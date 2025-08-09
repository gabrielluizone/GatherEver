# --------------------------------------------------------------------------
# Data Overseas - Sistema de Monitoramento Edge
# Proprietário e confidencial - Todos os direitos reservados
# Copyright (c) 2024 Data Overseas
# 
# Autor: Gabriel L S Silva
# Cargo: Data Scientist & Data Research
# Contato: gabriel.silva@dataoverseas.com.br
# 
# Este código contém segredos comerciais da Data Overseas.
# Cópia, distribuição ou modificação não autorizada é proibida.
# --------------------------------------------------------------------------

from typing import Dict, List, Optional, Any
from ultralytics.utils import LOGGER
from ultralytics import YOLOE
from ultralytics import YOLO

import numpy as np
import threading
import platform
import datetime
import requests
import logging
import hashlib
import base64
import psutil
import json
import time
import cv2
import os

class EdgeMonitoringCollector:
    """
    Sistema de Monitoramento Edge - Módulo Coletor
    Captura imagens, processa detecções e envia dados apenas quando há detecções.
    """
    
    def __init__(self, empresa_id: str, empresa_nome: str, coletor_id: str, 
                 coletor_nome: str, coletor_descricao: str, modelo: str,
                 camera_source: int = 0, confidence_threshold: float = 1,
                 slat: float = -23.960833, slon: float = -46.333889,
                 capture_interval: int = 15, server_key: str = "",
                 seeklist: list = list(), seekkey: str = "", retry_interval: int = 1800):  # 30 min em segundos
        """
        Inicializa o coletor de monitoramento edge.
        
        Args:
            empresa_id: ID da empresa
            empresa_nome: Nome da empresa
            coletor_id: ID do coletor
            coletor_nome: Nome do coletor
            coletor_descricao: Descrição do coletor
            modelo: Nome do arquivo do modelo (ex: "yoloe-11s-seg-pf.pt")
            camera_source: Fonte da câmera (0 para padrão)
            confidence_threshold: Limiar de confiança para detecções
            slat: Latitude fixa
            slon: Longitude fixa
            capture_interval: Intervalo entre capturas em segundos
            server_key: URL base do servidor (ex: "https://domain.ngrok-free.app")
            retry_interval: Intervalo para retry de conexão em segundos (padrão 30min)
        """
        
        # Configurações básicas
        self.EMPRESA_ID = empresa_id
        self.EMPRESA_NOME = empresa_nome
        self.COLETOR_ID = coletor_id
        self.COLETOR_NOME = coletor_nome
        self.COLETOR_DESCRICAO = coletor_descricao
        self.MODELO = modelo
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.CAMERA_SOURCE = camera_source
        self.SLAT = slat
        self.SLON = slon
        self.CAPTURE_INTERVAL = capture_interval
        self.RETRY_INTERVAL = retry_interval
        self.SEEKNOW = seeklist
        self.SEEKKEY = seekkey
        
        # URLs do servidor
        self.SERVER_URL = f'https://{server_key}.ngrok-free.app'
        self.ANALYSIS_URL = f"{self.SERVER_URL}/upload"
        self.HEALTH_URL = f"{self.SERVER_URL}/health"
        
        # Estado interno
        self.is_running = False
        self.model = None
        self.model_available = False
        self.is_yoloe = False
        self.last_server_check = 0
        self.server_available = False
        self.connection_failures = 0  # Contador de falhas consecutivas
        self.last_connection_log = 0  # Última vez que logamos problema de conexão
        
        # Setup logging
        self._setup_logging()
        
        # Criar diretório de modelos
        os.makedirs("models", exist_ok=True)
        
        # Inicializar modelo
        self._load_model()
        
    def _setup_logging(self):
        """Configura o logging do sistema."""
        # Criar diretório de logs se não existir
        os.makedirs("logs", exist_ok=True)
        
        # Configurar formatação
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configurar handler para arquivo com rotação
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            f'logs/collector_{self.COLETOR_ID}.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Configurar logger
        self.logger = logging.getLogger(f'EdgeCollector_{self.COLETOR_ID}')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        
        # Adicionar handler de console apenas se necessário (debug)
        if os.getenv('DEBUG', 'false').lower() == 'true':
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(console_handler)
    
    def _init_camera(self):
        """Inicializa a câmera com configurações otimizadas"""
        try:
            # Liberar câmera anterior se existir
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.release()
                time.sleep(1)  # Aguardar liberação completa
            
            self.camera = cv2.VideoCapture()
            
            if isinstance(self.CAMERA_SOURCE, str):
                # Para câmera IP
                camera_url = f"rtsp://admin:dataoverseas1@{self.CAMERA_SOURCE}"
                
                # Configurações mais agressivas para câmeras IP
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
                self.camera.set(cv2.CAP_PROP_FPS, 5)  # FPS baixo para evitar acúmulo
                
                success = self.camera.open(camera_url)
            else:
                # Para câmera local
                success = self.camera.open(self.CAMERA_SOURCE, cv2.CAP_V4L2)
                if success:
                    self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.camera.set(cv2.CAP_PROP_FPS, 10)  # FPS reduzido
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
            
            if not success or not self.camera.isOpened():
                self.logger.error("Failed to initialize camera")
                return False
            
            # Descartar frames iniciais para estabilização
            for _ in range(5):
                ret, _ = self.camera.read()
                if not ret:
                    break
                time.sleep(0.2)
            
            self.logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False
        
    def _load_model(self):
        """Carrega o modelo de detecção."""
        LOGGER.setLevel("ERROR")
        model_file = self.MODELO
        model_path = f"models/{model_file}"
        
        try:
            # Tentar carregar modelo existente primeiro
            if os.path.exists(model_path):
                try:
                    self.model = YOLO(model_path, verbose=False).to('cpu')
                    self.logger.info("ModeLe loaded successfully from local path")
                    self.model_available = True
                    return
                except Exception as e:
                    self.logger.error(f"Error loading existing model: {e}")
            
            # Se não disponível, tentar baixar
            try:
                # Tentar YOLO primeiro
                self.model = YOLO(model_file, verbose=False).to('cpu')
                self.logger.info("ModeLy loaded successfully from local path")
                self.model_available = True
            except Exception as e:
                self.logger.warning(f"ModeLy failed, attempting ModeLe: {e}")
                if YOLOE:
                    try:
                        self.model = YOLOE(model_file).to('cpu')
                        self.model_available = True
                        self.is_yoloe = True
                        self.logger.info("ModeLe model loaded successfully")
                    except Exception as e:
                        self.logger.error(f"ModeLe also failed: {e}")
                        raise Exception("Could not load any model")
                else:
                    raise Exception("ModeLe not available")
            
            # Mover para pasta de modelos se baixado
            if self.model_available and os.path.exists(model_file):
                os.rename(model_file, model_path)
                self.logger.info("ModeLy moved to models directory")
                
        except Exception as e:
            self.logger.error(f"Could not obtain any model: {e}")
            self.model_available = False
    
    def _capture_frame(self) -> Optional[Dict[str, Any]]:
        """Captura frame com limpeza agressiva de buffer"""
        try:
            # Verificar se câmera ainda está conectada
            if not hasattr(self, 'camera') or not self.camera.isOpened():
                self.logger.warning("Camera disconnected, attempting reconnection...")
                if not self._init_camera():
                    return None
            
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # PARA CÂMERAS IP: LIMPEZA MAIS AGRESSIVA DO BUFFER
                    if isinstance(self.CAMERA_SOURCE, str):
                        self.logger.info(f"🔄 Attempt {attempt + 1}: Aggressive buffer flush...")
                        
                        # Limpar buffer mais agressivamente
                        for i in range(10):  # Aumentado de 5 para 10
                            ret, _ = self.camera.read()
                            if not ret:
                                self.logger.warning(f"Buffer flush failed at frame {i}")
                                break
                            time.sleep(0.05)  # Delay menor entre leituras
                        
                        # Aguardar mais tempo para novo frame
                        time.sleep(1.5)  # Aumentado de 0.5s para 1.5s
                    
                    # Capturar frame final
                    ret, frame = self.camera.read()
                    if not ret:
                        self.logger.error(f"Failed to capture frame on attempt {attempt + 1}")
                        if attempt < max_attempts - 1:
                            # Tentar reconectar na próxima tentativa
                            self.logger.info("Attempting camera reconnection...")
                            self.camera.release()
                            time.sleep(2)
                            if not self._init_camera():
                                continue
                        else:
                            return None
                        continue
                    
                    # Verificar se frame não está vazio ou corrompido
                    if frame is None or frame.size == 0:
                        self.logger.warning(f"Empty frame on attempt {attempt + 1}")
                        continue
                    
                    # Calcular hash do frame
                    current_time = datetime.datetime.now()
                    timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
                    
                    # Verificar se é o mesmo frame anterior
                    if hasattr(self, 'last_frame_hash'):
                        if frame_hash == self.last_frame_hash:
                            self.logger.warning(f"⚠️ Duplicate frame detected! Hash: {frame_hash} (attempt {attempt + 1})")
                            
                            if attempt < max_attempts - 1:
                                # Tentar forçar novo frame
                                time.sleep(2)
                                continue
                            else:
                                # Se é a última tentativa, usar o frame mesmo sendo duplicado
                                self.logger.error("All attempts resulted in duplicate frames - using anyway")
                    
                    self.last_frame_hash = frame_hash
                    self.logger.info(f"✅ Fresh frame captured: {timestamp_str} - Hash: {frame_hash}")
                    
                    # Encoded image with compression
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                    _, buffer = cv2.imencode('.jpg', frame, encode_params)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    camera_info = {
                        'available': True,
                        'width': frame.shape[1],
                        'height': frame.shape[0],
                        'fps': self.camera.get(cv2.CAP_PROP_FPS),
                        'format': self.camera.get(cv2.CAP_PROP_FORMAT),
                        'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                        'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
                        'saturation': self.camera.get(cv2.CAP_PROP_SATURATION),
                        'hue': self.camera.get(cv2.CAP_PROP_HUE),
                        'gain': self.camera.get(cv2.CAP_PROP_GAIN),
                        'exposure': self.camera.get(cv2.CAP_PROP_EXPOSURE),
                        'timestamp': timestamp_str,
                        'frame_hash': frame_hash,
                        'attempt': attempt + 1,
                        'frame': frame,
                        'frame_base64': frame_base64
                    }
                    
                    return camera_info
                    
                except Exception as e:
                    self.logger.error(f"Error on capture attempt {attempt + 1}: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(1)
                        continue
                    else:
                        return None
            
            return None
                
        except Exception as e:
            self.logger.error(f"Camera capture error: {e}")
            return None

    def _run_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Executa detecção no frame.
        
        Args:
            frame: Frame da imagem
            
        Returns:
            Lista de detecções encontradas
        """
        if not self.model_available:
            return []
            
        detections = []
        
        try:
            results = self.model(frame, conf=self.CONFIDENCE_THRESHOLD)
            
            for result in results:
                if self.is_yoloe:
                    # YOLOE segmentation - converter máscaras para caixas
                    if hasattr(result, 'masks') and result.masks is not None:
                        for mask in result.masks:
                            x, y, w, h = cv2.boundingRect(mask.data[0].cpu().numpy().astype(np.uint8))
                            confidence = mask.conf[0].cpu().numpy() if hasattr(mask, 'conf') else 0.8
                            class_id = int(mask.cls[0].cpu().numpy()) if hasattr(mask, 'cls') else 0
                            class_name = self.model.names[class_id] if hasattr(self.model, 'names') else str(class_id)
                            
                            detections.append({
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "confidence": float(confidence),
                                "class": class_name,
                                "class_id": class_id
                            })
                else:
                    # Detecção YOLO padrão
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[class_id]
                            
                            detections.append({
                                "x": int(x1),
                                "y": int(y1),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1),
                                "confidence": float(confidence),
                                "class": class_name,
                                "class_id": class_id
                            })
            
            self.logger.info(f"Detections found: {len(detections)}")
            return detections
                            
        except Exception as e:
            self.logger.error(f"Detection processing error: {e}")
            return []
    
    def check_for_duplicate_instances(self):
        """Verifica se há múltiplas instâncias rodando"""
        import psutil
        
        current_pid = os.getpid()
        python_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    if proc.info['cmdline'] and any('EdgeMonitoringCollector' in str(cmd) for cmd in proc.info['cmdline']):
                        python_processes.append(proc.info['pid'])
            except:
                continue
        
        if len(python_processes) > 1:
            self.logger.warning(f"⚠️ Multiple instances detected: PIDs {python_processes}")
            self.logger.warning(f"Current PID: {current_pid}")
            return True
        
        return False
    
    # SOLUÇÃO ALTERNATIVA: Reconectar câmera a cada captura (para câmeras problemáticas)
    def _capture_frame_reconnect(self) -> Optional[Dict[str, Any]]:
        """Método alternativo: reconectar câmera a cada captura (mais lento mas mais confiável)"""
        try:
            # Fechar conexão anterior SEMPRE
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.release()
                time.sleep(1)  # Aguardar liberação completa
            
            # Nova conexão
            self.camera = cv2.VideoCapture()
            
            if isinstance(self.CAMERA_SOURCE, str):
                camera_url = f"rtsp://admin:dataoverseas1@{self.CAMERA_SOURCE}"
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                success = self.camera.open(camera_url)
            else:
                success = self.camera.open(self.CAMERA_SOURCE, cv2.CAP_V4L2)
                if success:
                    self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not success or not self.camera.isOpened():
                self.logger.error("Failed to reconnect camera")
                return None
            
            # Aguardar estabilização mais tempo
            time.sleep(2)
            
            # Descartar alguns frames iniciais
            for _ in range(5):
                ret, _ = self.camera.read()
                if not ret:
                    break
                time.sleep(0.1)
            
            # Capturar frame final
            ret, frame = self.camera.read()
            if not ret or frame is None or frame.size == 0:
                self.logger.error("Failed to capture frame after reconnection")
                return None
            
            current_time = datetime.datetime.now()
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
            
            self.logger.info(f"📷 Fresh reconnection - Frame: {timestamp_str} - Hash: {frame_hash}")
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            camera_info = {
                'available': True,
                'width': frame.shape[1],
                'height': frame.shape[0],
                'timestamp': timestamp_str,
                'frame_hash': frame_hash,
                'reconnect_method': True,
                'frame': frame,
                'frame_base64': frame_base64
            }
            
            return camera_info
            
        except Exception as e:
            self.logger.error(f"Camera reconnect capture error: {e}")
            return None

    def _load_image_from_path(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Carrega imagem de um arquivo local.
        
        Args:
            image_path: Caminho para o arquivo de imagem
            
        Returns:
            Informações da imagem carregada ou None se falhar
        """
        try:
            self.logger.info(f"📁 Loading image from file: {image_path}")
            
            if not os.path.exists(image_path):
                self.logger.error(f"❌ File not found: {image_path}")
                return None
            
            # Carregar imagem
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"❌ Failed to load image: {image_path}")
                return None
            
            # Informações da imagem
            current_time = datetime.datetime.now()
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_hash = hashlib.md5(image.tobytes()).hexdigest()[:12]
            
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            size_mb = image.nbytes / (1024 * 1024)
            
            # Codificar em base64
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            _, buffer = cv2.imencode('.jpg', image, encode_params)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            encoded_size_mb = len(image_base64) * 3/4 / (1024*1024)
            
            self.logger.info(f"✅ Image loaded successfully!")
            self.logger.info(f"   📐 Resolution: {width}x{height}")
            self.logger.info(f"   💾 Size: {size_mb:.2f} MB (raw)")
            self.logger.info(f"   📤 Size: {encoded_size_mb:.2f} MB (encoded)")
            self.logger.info(f"   🔑 Hash: {image_hash}")
            
            image_info = {
                'available': True,
                'width': width,
                'height': height,
                'channels': channels,
                'raw_size_mb': size_mb,
                'encoded_size_mb': encoded_size_mb,
                'timestamp': timestamp_str,
                'frame_hash': image_hash,
                'photo_hash': image_hash,
                'is_duplicate': False,
                'source_type': 'file',
                'source_path': image_path,
                'frame': image,
                'frame_base64': image_base64
            }
            
            return image_info
            
        except Exception as e:
            self.logger.error(f"❌ Error loading image from file: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def test_with_image(self, image_source: str, send_to_server: bool = True) -> Dict[str, Any]:
        """
        Testa o sistema completo usando uma imagem específica.
        
        Args:
            image_source: Caminho do arquivo ou URL da imagem
            send_to_server: Se deve enviar para o servidor (padrão True)
            
        Returns:
            Resultado do teste com estatísticas
        """
        try:
            self.logger.info("🧪 STARTING IMAGE TEST...")
            self.logger.info(f"   📍 Source: {image_source}")
            self.logger.info(f"   📤 Send to server: {send_to_server}")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            # 1. CARREGAR IMAGEM
            if image_source.startswith(('http://', 'https://')):
                image_info = self._load_image_from_url(image_source)
            else:
                image_info = self._load_image_from_path(image_source)
            
            if not image_info:
                self.logger.error("❌ Failed to load image")
                return {'success': False, 'error': 'Failed to load image'}
            
            load_time = time.time() - start_time
            
            # 2. EXECUTAR DETECÇÃO
            self.logger.info("🔍 Running AI detection on test image...")
            detection_start = time.time()
            
            detections = self._run_detection(image_info['frame'])
            
            detection_time = time.time() - detection_start
            self.logger.info(f"🎯 Detection completed: {len(detections)} objects found in {detection_time:.2f}s")
            
            # Log das detecções encontradas
            if detections:
                self.logger.info("📋 DETECTIONS FOUND:")
                for i, det in enumerate(detections):
                    self.logger.info(f"   {i+1}. {det['class']} - {det['confidence']:.1%} confidence")
            else:
                self.logger.info("🔍 No objects detected in test image")
            
            # 3. PROCESSAR IMAGEM (apenas se houver detecções)
            processed_image_base64 = ""
            processing_time = 0
            
            if detections:
                self.logger.info("🎨 Processing image with detection annotations...")
                processing_start = time.time()
                
                processed_image_base64 = self._process_image_with_detections(
                    image_info['frame'], detections
                )
                
                processing_time = time.time() - processing_start
                self.logger.info(f"✅ Image processing completed in {processing_time:.2f}s")
            else:
                self.logger.info("⏭️ Skipping image processing (no detections)")
            
            # 4. CRIAR PAYLOAD
            payload = None
            payload_time = 0
            
            if detections:
                self.logger.info("📦 Creating payload...")
                payload_start = time.time()
                
                payload = self._create_payload(image_info, detections, processed_image_base64)
                
                payload_time = time.time() - payload_start
                self.logger.info(f"✅ Payload created in {payload_time:.2f}s")
            
            # 5. ENVIAR PARA SERVIDOR (se solicitado e há detecções)
            server_sent = False
            server_time = 0
            
            if send_to_server and detections and self.SERVER_URL:
                self.logger.info("📡 Testing server connection...")
                
                if not self.server_available:
                    self._check_server_connection()
                
                if self.server_available and payload:
                    self.logger.info("📤 Sending test data to server...")
                    server_start = time.time()
                    
                    server_sent = self._send_to_server(payload)
                    
                    server_time = time.time() - server_start
                    
                    if server_sent:
                        self.logger.info(f"✅ Server transmission successful in {server_time:.2f}s")
                    else:
                        self.logger.warning(f"❌ Server transmission failed after {server_time:.2f}s")
                else:
                    self.logger.warning("❌ Server not available or no payload to send")
            elif not send_to_server:
                self.logger.info("⏭️ Skipping server transmission (not requested)")
            elif not detections:
                self.logger.info("⏭️ Skipping server transmission (no detections)")
            elif not self.SERVER_URL:
                self.logger.info("⏭️ Skipping server transmission (offline mode)")
            
            # ESTATÍSTICAS FINAIS
            total_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("📊 TEST RESULTS SUMMARY:")
            self.logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
            self.logger.info(f"   📁 Load time: {load_time:.2f}s")
            self.logger.info(f"   🔍 Detection time: {detection_time:.2f}s")
            self.logger.info(f"   🎨 Processing time: {processing_time:.2f}s")
            self.logger.info(f"   📦 Payload time: {payload_time:.2f}s")
            self.logger.info(f"   📤 Server time: {server_time:.2f}s")
            self.logger.info(f"   🎯 Objects detected: {len(detections)}")
            self.logger.info(f"   📡 Server sent: {'✅ YES' if server_sent else '❌ NO'}")
            
            # Limpar memória
            try:
                if 'frame' in image_info:
                    del image_info['frame']
                if payload:
                    del payload
            except:
                pass
            
            result = {
                'success': True,
                'image_loaded': bool(image_info),
                'detections_count': len(detections),
                'detections': detections,
                'server_sent': server_sent,
                'timings': {
                    'total': total_time,
                    'load': load_time,
                    'detection': detection_time,
                    'processing': processing_time,
                    'payload': payload_time,
                    'server': server_time
                },
                'image_info': {
                    'width': image_info['width'],
                    'height': image_info['height'],
                    'size_mb': image_info['raw_size_mb'],
                    'hash': image_info['photo_hash'],
                    'source': image_source
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Test failed with error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
        
    def test_batch_images(self, image_sources: List[str], send_to_server: bool = True, interval: float = 2.0) -> Dict[str, Any]:
        """
        Testa múltiplas imagens em sequência.
        
        Args:
            image_sources: Lista de caminhos/URLs das imagens
            send_to_server: Se deve enviar para o servidor
            interval: Intervalo entre testes em segundos
            
        Returns:
            Resultado consolidado dos testes
        """
        try:
            self.logger.info(f"🧪 STARTING BATCH TEST with {len(image_sources)} images")
            self.logger.info(f"   📤 Send to server: {send_to_server}")
            self.logger.info(f"   ⏳ Interval: {interval}s")
            self.logger.info("="*60)
            
            results = []
            total_detections = 0
            successful_tests = 0
            total_start = time.time()
            
            for i, image_source in enumerate(image_sources):
                self.logger.info(f"📷 BATCH TEST {i+1}/{len(image_sources)}: {os.path.basename(image_source)}")
                
                result = self.test_with_image(image_source, send_to_server)
                results.append({
                    'index': i + 1,
                    'source': image_source,
                    'result': result
                })
                
                if result['success']:
                    successful_tests += 1
                    total_detections += result['detections_count']
                    self.logger.info(f"   ✅ Test {i+1} SUCCESS: {result['detections_count']} detections")
                else:
                    self.logger.error(f"   ❌ Test {i+1} FAILED: {result.get('error', 'Unknown error')}")
                
                # Intervalo entre testes (exceto no último)
                if i < len(image_sources) - 1:
                    self.logger.info(f"   ⏳ Waiting {interval}s until next test...")
                    time.sleep(interval)
            
            total_time = time.time() - total_start
            
            # ESTATÍSTICAS CONSOLIDADAS
            self.logger.info("="*60)
            self.logger.info("📊 BATCH TEST SUMMARY:")
            self.logger.info(f"   📷 Images tested: {len(image_sources)}")
            self.logger.info(f"   ✅ Successful tests: {successful_tests}")
            self.logger.info(f"   ❌ Failed tests: {len(image_sources) - successful_tests}")
            self.logger.info(f"   🎯 Total detections: {total_detections}")
            self.logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
            self.logger.info(f"   ⚡ Avg time per image: {total_time/len(image_sources):.2f}s")
            
            batch_result = {
                'success': successful_tests > 0,
                'total_images': len(image_sources),
                'successful_tests': successful_tests,
                'failed_tests': len(image_sources) - successful_tests,
                'total_detections': total_detections,
                'total_time': total_time,
                'avg_time_per_image': total_time / len(image_sources),
                'detailed_results': results
            }
            
            return batch_result
            
        except Exception as e:
            self.logger.error(f"❌ Batch test failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}

    def test_directory_images(self, directory_path: str, image_extensions: List[str] = None, 
                            send_to_server: bool = True, interval: float = 2.0) -> Dict[str, Any]:
        """
        Testa todas as imagens de um diretório.
        
        Args:
            directory_path: Caminho do diretório
            image_extensions: Extensões aceitas (padrão: jpg, jpeg, png, bmp)
            send_to_server: Se deve enviar para o servidor
            interval: Intervalo entre testes
            
        Returns:
            Resultado dos testes
        """
        try:
            if image_extensions is None:
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            
            self.logger.info(f"📁 SCANNING directory: {directory_path}")
            
            if not os.path.exists(directory_path):
                self.logger.error(f"❌ Directory not found: {directory_path}")
                return {'success': False, 'error': 'Directory not found'}
            
            # Encontrar todas as imagens
            image_files = []
            for file in os.listdir(directory_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    image_files.append(os.path.join(directory_path, file))
            
            if not image_files:
                self.logger.warning(f"⚠️ No images found in directory with extensions: {image_extensions}")
                return {'success': False, 'error': 'No images found'}
            
            self.logger.info(f"📷 Found {len(image_files)} images to test")
            
            # Ordenar arquivos para teste consistente
            image_files.sort()
            
            # Executar teste em lote
            return self.test_batch_images(image_files, send_to_server, interval)
            
        except Exception as e:
            self.logger.error(f"❌ Directory test failed: {e}")
            return {'success': False, 'error': str(e)}

    def quick_test_image(self, image_source: str) -> bool:
        """
        Teste rápido de uma imagem (sem enviar para servidor).
        
        Args:
            image_source: Caminho ou URL da imagem
            
        Returns:
            True se o teste foi bem-sucedido
        """
        self.logger.info(f"⚡ QUICK TEST: {os.path.basename(image_source)}")
        
        result = self.test_with_image(image_source, send_to_server=False)
        
        if result['success']:
            detections = result['detections_count']
            total_time = result['timings']['total']
            self.logger.info(f"   ✅ SUCCESS: {detections} detections in {total_time:.2f}s")
            return True
        else:
            self.logger.error(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
            return False

    def _load_image_from_url(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Baixa e carrega imagem de uma URL.
        
        Args:
            image_url: URL da imagem
            
        Returns:
            Informações da imagem carregada ou None se falhar
        """
        try:
            self.logger.info(f"🌐 Downloading image from URL: {image_url}")
            
            # Baixar imagem
            headers = {
                'User-Agent': 'EdgeCollector/1.0',
                'Accept': 'image/*'
            }
            
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Converter bytes para numpy array
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                self.logger.error(f"❌ Failed to decode image from URL: {image_url}")
                return None
            
            # Informações da imagem
            current_time = datetime.datetime.now()
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_hash = hashlib.md5(image.tobytes()).hexdigest()[:12]
            
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            size_mb = image.nbytes / (1024 * 1024)
            
            # Codificar em base64
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            _, buffer = cv2.imencode('.jpg', image, encode_params)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            encoded_size_mb = len(image_base64) * 3/4 / (1024*1024)
            
            self.logger.info(f"✅ Image downloaded successfully!")
            self.logger.info(f"   📐 Resolution: {width}x{height}")
            self.logger.info(f"   💾 Size: {size_mb:.2f} MB (raw)")
            self.logger.info(f"   📤 Size: {encoded_size_mb:.2f} MB (encoded)")
            self.logger.info(f"   🔑 Hash: {image_hash}")
            
            image_info = {
                'available': True,
                'width': width,
                'height': height,
                'channels': channels,
                'raw_size_mb': size_mb,
                'encoded_size_mb': encoded_size_mb,
                'timestamp': timestamp_str,
                'frame_hash': image_hash,
                'photo_hash': image_hash,
                'is_duplicate': False,
                'source_type': 'url',
                'source_url': image_url,
                'download_size_bytes': len(response.content),
                'frame': image,
                'frame_base64': image_base64
            }
            
            return image_info
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Error downloading image: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error processing downloaded image: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _process_image_with_detections(self, frame: np.ndarray, detections: List[Dict]) -> str:
        """
        Processa a imagem desenhando as detecções com tratamento robusto de erros.
        """
        try:
            if frame is None or frame.size == 0:
                self.logger.error("❌ Invalid frame for processing")
                return ""
                
            img = frame.copy()
            text_rects = []
            
            # Verificar se há detecções para processar
            if not detections or len(detections) == 0:
                self.logger.info("No detections to draw")
                # Codificar imagem sem modificações
                _, buffer = cv2.imencode('.jpg', img)
                return base64.b64encode(buffer).decode('utf-8')
            
            # Desenhar caixas de detecção com verificações adicionais
            for i, detection in enumerate(detections):
                try:
                    # Validar campos obrigatórios
                    required_fields = ['x', 'y', 'width', 'height', 'confidence', 'class', 'class_id']
                    if not all(field in detection for field in required_fields):
                        self.logger.warning(f"Detection {i} missing required fields, skipping")
                        continue
                    
                    x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
                    confidence = detection['confidence']
                    class_name = detection['class']
                    
                    # Validar coordenadas
                    img_h, img_w = img.shape[:2]
                    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                        self.logger.warning(f"Detection {i} coordinates out of bounds, adjusting")
                        x = max(0, min(x, img_w - 1))
                        y = max(0, min(y, img_h - 1))
                        w = min(w, img_w - x)
                        h = min(h, img_h - y)
                    
                    # Gerar cor única baseada no class_id
                    class_id = detection.get('class_id', 0)
                    color_hash = hashlib.sha256(str(class_id).encode()).hexdigest()
                    color = (
                        int(color_hash[:2], 16),
                        int(color_hash[2:4], 16), 
                        int(color_hash[4:6], 16)
                    )
                    
                    # Desenhar retângulo com verificações
                    if w > 0 and h > 0:
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        
                        # Preparar texto
                        text = f"{class_name} {int(confidence*100)}%"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        
                        # Calcular posição do texto
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        text_x = max(0, min(x, img_w - text_width))
                        text_y = max(text_height, y - 5)
                        
                        # Desenhar fundo do texto
                        cv2.rectangle(img, 
                                    (text_x - 2, text_y - text_height - 2),
                                    (text_x + text_width + 2, text_y + 2),
                                    color, -1)
                        
                        # Desenhar texto
                        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                        
                except Exception as det_error:
                    self.logger.error(f"Error processing detection {i}: {det_error}")
                    continue
            
            # Codificar imagem processada
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            success, buffer = cv2.imencode('.jpg', img, encode_params)
            
            if not success:
                self.logger.error("Failed to encode processed image")
                return ""
                
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Critical error in image processing: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""
    
    def _create_payload(self, camera_info: Dict, detections: List[Dict], processed_image_base64: str) -> Dict:
        """
        Cria o payload para envio ao servidor.
        
        Args:
            camera_info: Informações da câmera
            detections: Lista de detecções
            processed_image_base64: Imagem processada em base64
            
        Returns:
            Payload formatado para envio
        """
        modelo_nome = os.path.splitext(os.path.basename(self.MODELO))[0]
        
        # Calcular tamanhos das imagens em MB
        original_img_size_mb = len(camera_info['frame_base64']) * 3/4 / (1024*1024)
        processed_img_size_mb = len(processed_image_base64) * 3/4 / (1024*1024) if processed_image_base64 else 0
        
        payload = {
            "image_data": {
                "original": {
                    "base64": camera_info['frame_base64'],
                    "size_mb": round(original_img_size_mb, 2),
                    "filename": f"RAW-{self.EMPRESA_ID}-{self.COLETOR_ID}-{modelo_nome}-{camera_info['timestamp']}.jpg"
                },
                "processed": {
                    "base64": processed_image_base64,
                    "size_mb": round(processed_img_size_mb, 2),
                    "filename": f"PRC-{self.EMPRESA_ID}-{self.COLETOR_ID}-{modelo_nome}-{camera_info['timestamp']}.jpg"
                }
            },
            
            "company_info": {
                "id": self.EMPRESA_ID,
                "name": self.EMPRESA_NOME,
                "collector": {
                    "id": self.COLETOR_ID,
                    "name": self.COLETOR_NOME,
                    "model": modelo_nome,
                    "description": self.COLETOR_DESCRICAO
                },
                "fixed_location": {
                    "latitude": self.SLAT,
                    "longitude": self.SLON
                }
            },
            
            "detections": {
                "data": detections,
                "count": len(detections),
                "confidence_threshold": self.CONFIDENCE_THRESHOLD
            },
            
            "system_metadata": {
                "camera": {
                    "resolution": {
                        "width": camera_info['width'],
                        "height": camera_info['height']
                    },
                    "settings": {
                        "fps": camera_info.get('fps', 0.0),
                        "format": camera_info.get('format', 0.0),
                        "brightness": camera_info.get('brightness', 0.0),
                        "contrast": camera_info.get('contrast', 0.0),
                        "saturation": camera_info.get('saturation', 0.0),
                        "hue": camera_info.get('hue', 0.0),
                        "gain": camera_info.get('gain', 0.0),
                        "exposure": camera_info.get('exposure', 0.0)
                    }
                },
                "os": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version()
                },
                "hardware": {
                    "architecture": platform.machine(),
                    "cpu": {
                        "physical_cores": psutil.cpu_count(logical=False),
                        "logical_cores": psutil.cpu_count(logical=True),
                        "max_frequency": psutil.cpu_freq().max if hasattr(psutil, 'cpu_freq') else None,
                    }
                },
                "timestamp": camera_info.get('timestamp')
            },
            
            "system_status": {
                "cpu": psutil.cpu_percent(interval=0.3),
                "memory": psutil.virtual_memory().percent,
                "uptime": int(datetime.datetime.now().timestamp() - psutil.boot_time())
            },

            "seek_on_image": {
                "user_prompt": self.SEEKNOW,
                "prompt_title": self.SEEKKEY
            }
        }
        
        return payload
    
    def _check_server_connection(self) -> bool:
        """Verifica se o servidor está disponível com timeout mais curto."""
        if not self.SERVER_URL:
            return False
            
        try:
            # Timeout mais curto para não bloquear muito tempo
            response = requests.get(self.HEALTH_URL, timeout=5)
            if response.status_code == 200:
                if self.connection_failures > 0:
                    self.logger.info(f"Server connection restored after {self.connection_failures} failures")
                    self.connection_failures = 0
                
                self.server_available = True
                self.last_server_check = time.time()
                return True
            else:
                self.logger.warning(f"Server health check failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            self.logger.warning("Server connection timeout")
        except Exception as e:
            self.connection_failures += 1
            current_time = time.time()
            
            # Log com menos frequência para não spammar
            if self.connection_failures == 1 or (current_time - self.last_connection_log > 600):  # 10 min
                self.logger.error(f"Server connection failed (attempt #{self.connection_failures}): {e}")
                self.last_connection_log = current_time
        
        self.server_available = False
        return False
    
    def _send_to_server(self, payload: Dict) -> bool:
        """
        Envia payload para o servidor.
        
        Args:
            payload: Dados para envio
            
        Returns:
            True se enviado com sucesso, False caso contrário
        """
        # Sempre tentar verificar conexão se não está disponível
        if not self.server_available:
            current_time = time.time()
            # Apenas fazer retry se passou o tempo necessário OU se nunca testamos
            if (current_time - self.last_server_check < self.RETRY_INTERVAL) and self.last_server_check > 0:
                return False
            
            if not self._check_server_connection():
                return False
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': f'EdgeCollector/{self.COLETOR_ID}',
                'Connection': 'close'
            }
            
            response = requests.post(
                self.ANALYSIS_URL,
                json=payload,
                headers=headers,
                timeout=30,  # Timeout menor - servidor responde rápido agora
                stream=False
            )
            
            # ACEITAR TANTO 200 (OK) QUANTO 202 (ACCEPTED)
            if response.status_code in [200, 202]:
                response_data = response.json()
                if response.status_code == 202:
                    self.logger.info(f"Data accepted for processing: {response_data.get('message', '')}")
                else:
                    self.logger.info("Data processed successfully")
                
                self.connection_failures = 0
                self.server_available = True
                return True
            else:
                self.logger.warning(f"Server responded with status {response.status_code}")
                self.server_available = False
                self.connection_failures += 1
                return False
                
        except requests.exceptions.Timeout:
            self.logger.warning("Server response timeout - unusual for async processing")
            self.server_available = False
            self.connection_failures += 1
            return False
        except Exception as e:
            self.logger.error(f"Error sending data: {e}")
            if "connection" in str(e).lower():
                self.server_available = False
            self.connection_failures += 1
            return False
        
    def _monitoring_loop(self):
        """Loop principal de monitoramento fotográfico com tratamento robusto de erros."""
        self.logger.info(f"📷 Starting PHOTOGRAPHIC monitoring system")
        self.logger.info(f"   📊 Interval: {self.CAPTURE_INTERVAL}s between photos")
        self.logger.info(f"   🎯 Mode: High-resolution photo snapshots")
        self.logger.info("="*60)
        
        # Verificar conexão inicial do servidor
        if self.SERVER_URL:
            try:
                self._check_server_connection()
            except Exception as e:
                self.logger.error(f"Error checking initial server connection: {e}")
        
        consecutive_errors = 0
        max_consecutive_errors = 3
        photo_count = 0
        
        while self.is_running:
            photo_info = None
            processed_image_base64 = ""
            payload = None
            
            try:
                photo_count += 1
                start_time = time.time()
                
                self.logger.info(f"📷 PHOTO SESSION #{photo_count} - {datetime.datetime.now()}")
                self.logger.info("="*40)
                
                # TIRAR FOTO
                try:
                    photo_info = self._take_photo_snapshot()
                except Exception as photo_error:
                    self.logger.error(f"Photo capture exception: {photo_error}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    photo_info = None
                
                if not photo_info:
                    self.logger.warning("❌ Photo capture failed, will retry next session")
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error(f"💥 Too many photo failures ({consecutive_errors}), extended wait")
                        time.sleep(120)  # 2 minutos de pausa
                        consecutive_errors = 0
                    else:
                        time.sleep(30)  # 30 segundos para retry
                    
                    continue
                
                # Reset contador após sucesso
                consecutive_errors = 0
                
                if photo_info.get('is_duplicate', False):
                    self.logger.warning("⚠️ Duplicate photo detected but continuing...")
                
                self.logger.info("✅ PHOTO CAPTURED SUCCESSFULLY")
                
                # Executar detecção na foto
                detections = []
                try:
                    self.logger.info("🔍 Running AI detection on photo...")
                    detections = self._run_detection(photo_info['frame'])
                    self.logger.info(f"🎯 Detection results: {len(detections)} objects found")
                except Exception as detection_error:
                    self.logger.error(f"Detection error: {detection_error}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    detections = []
                
                # Processar se houver detecções
                if detections:
                    try:
                        self.logger.info("🎨 Processing photo with detection annotations...")
                        processed_image_base64 = self._process_image_with_detections(
                            photo_info['frame'], detections
                        )
                        
                        if not processed_image_base64:
                            self.logger.warning("Failed to process image, using original")
                            # Usar imagem original se processamento falhar
                            processed_image_base64 = photo_info.get('frame_base64', '')
                            
                    except Exception as processing_error:
                        self.logger.error(f"Image processing error: {processing_error}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        # Usar imagem original como fallback
                        processed_image_base64 = photo_info.get('frame_base64', '')
                    
                    # Remover foto da memória para economizar RAM
                    try:
                        if 'frame' in photo_info:
                            del photo_info['frame']
                    except Exception as cleanup_error:
                        self.logger.warning(f"Memory cleanup warning: {cleanup_error}")
                    
                    # Criar payload
                    try:
                        payload = self._create_payload(photo_info, detections, processed_image_base64)
                    except Exception as payload_error:
                        self.logger.error(f"Payload creation error: {payload_error}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        payload = None
                    
                    # Tentar enviar para servidor
                    if self.SERVER_URL and payload:
                        try:
                            if not self.server_available:
                                self.logger.info("📡 Checking server connection...")
                                self._check_server_connection()
                            
                            if self.server_available:
                                success = self._send_to_server(payload)
                                if success:
                                    self.logger.info(f"📤 SUCCESS: Photo with {len(detections)} detections sent!")
                                else:
                                    self.logger.warning(f"📤 FAILED: Photo transmission failed")
                            else:
                                self.logger.warning(f"📡 Server offline: Photo processed locally only")
                                
                        except Exception as send_error:
                            self.logger.error(f"Server communication error: {send_error}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            
                    elif not self.SERVER_URL:
                        self.logger.info(f"💾 Offline mode: Photo with {len(detections)} detections processed")
                    else:
                        self.logger.warning("Cannot send data: no valid payload created")
                else:
                    self.logger.info("🔍 No objects detected in photo - skipping transmission")
                    # Limpar memória mesmo sem detecções
                    try:
                        if photo_info and 'frame' in photo_info:
                            del photo_info['frame']
                    except:
                        pass
                
                # Calcular tempo de espera até próxima foto
                processing_time = time.time() - start_time
                sleep_time = max(10, self.CAPTURE_INTERVAL - processing_time)  # Mínimo 10s entre fotos
                
                self.logger.info(f"⏱️ Photo session #{photo_count} completed in {processing_time:.2f}s")
                self.logger.info(f"💤 Waiting {sleep_time:.1f}s until next photo session...")
                self.logger.info("="*60)
                
                # Sleep com verificação de interrupção
                sleep_start = time.time()
                while self.is_running and (time.time() - sleep_start) < sleep_time:
                    time.sleep(min(1, sleep_time - (time.time() - sleep_start)))
                        
            except KeyboardInterrupt:
                self.logger.info("⌨️ User interruption - stopping photo sessions")
                break
            except Exception as e:
                self.logger.error(f"💥 CRITICAL ERROR in photo session #{photo_count}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                
                consecutive_errors += 1
                
                # Cleanup de emergência
                try:
                    if photo_info and 'frame' in photo_info:
                        del photo_info['frame']
                    if 'processed_image_base64' in locals():
                        del processed_image_base64
                    if payload:
                        del payload
                except:
                    pass
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(f"💥 Too many critical errors ({consecutive_errors}), extended pause")
                    time.sleep(120)
                    consecutive_errors = 0
                else:
                    self.logger.info(f"Waiting 30s before retry (error #{consecutive_errors})")
                    time.sleep(30)
                
                continue
        
        self.logger.info("📷 Photo monitoring sessions ended")

    def diagnose_system(self):
        """Executa diagnóstico completo do sistema"""
        self.logger.info("🔍 STARTING SYSTEM DIAGNOSIS...")
        
        # 1. Verificar modelo
        self.logger.info(f"📊 Model available: {self.model_available}")
        if self.model_available:
            self.logger.info(f"   Model type: {'YOLOE' if self.is_yoloe else 'YOLO'}")
            self.logger.info(f"   Model file: {self.MODELO}")
        
        # 2. Verificar câmera
        self.logger.info("📷 Testing camera connection...")
        camera_ok = self.test_photo_capture()
        
        # 3. Verificar servidor
        if self.SERVER_URL:
            self.logger.info("🌐 Testing server connection...")
            server_ok = self._check_server_connection()
            self.logger.info(f"   Server available: {server_ok}")
        else:
            self.logger.info("🌐 Running in offline mode (no server configured)")
            server_ok = None
        
        # 4. Verificar recursos do sistema
        self.logger.info("💻 System resources:")
        self.logger.info(f"   CPU usage: {psutil.cpu_percent()}%")
        self.logger.info(f"   Memory usage: {psutil.virtual_memory().percent}%")
        self.logger.info(f"   Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        # 5. Verificar processos duplicados
        duplicate_processes = self.check_for_duplicate_instances()
        self.logger.info(f"🔄 Duplicate instances: {duplicate_processes}")
        
        # Resumo
        self.logger.info("📋 DIAGNOSIS SUMMARY:")
        self.logger.info(f"   ✅ Model: {'OK' if self.model_available else '❌ FAILED'}")
        self.logger.info(f"   ✅ Camera: {'OK' if camera_ok else '❌ FAILED'}")
        self.logger.info(f"   ✅ Server: {'OK' if server_ok else ('OFFLINE' if server_ok is False else 'N/A')}")
        self.logger.info(f"   ✅ Resources: OK")
        self.logger.info(f"   ✅ Duplicates: {'⚠️ YES' if duplicate_processes else 'OK'}")
        
        return {
            'model_ok': self.model_available,
            'camera_ok': camera_ok,
            'server_ok': server_ok,
            'duplicate_processes': duplicate_processes
        }
    
    def _take_photo_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Tira uma FOTO única em máxima resolução - como uma câmera fotográfica
        Não é para streaming, é para captura única de alta qualidade
        """
        camera = None
        try:
            self.logger.info("📷 TAKING PHOTO - Opening camera for snapshot...")
            
            # ABRIR CÂMERA APENAS PARA FOTO
            camera = cv2.VideoCapture()
            
            if isinstance(self.CAMERA_SOURCE, str):
                # Câmera IP
                camera_url = f"rtsp://admin:dataoverseas1@{self.CAMERA_SOURCE}"
                self.logger.info(f"Connecting to IP camera for photo: {camera_url}")
                success = camera.open(camera_url)
            else:
                # Câmera USB
                self.logger.info(f"Connecting to USB camera for photo: {self.CAMERA_SOURCE}")
                success = camera.open(self.CAMERA_SOURCE)
            
            if not success or not camera.isOpened():
                self.logger.error("❌ Failed to open camera for photo")
                return None
            
            # CONFIGURAR PARA MÁXIMA QUALIDADE FOTOGRÁFICA
            self.logger.info("🎯 Setting camera to PHOTO mode (maximum resolution)...")
            
            if isinstance(self.CAMERA_SOURCE, str):
                # Para câmera IP - configurações para foto
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Tentar definir resolução máxima para IP
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                # Para câmera USB - configurações fotográficas
                # MÁXIMA RESOLUÇÃO POSSÍVEL
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Full HD width
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Full HD height
                
                # Se não conseguir Full HD, tentar HD
                if camera.get(cv2.CAP_PROP_FRAME_WIDTH) < 1920:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Configurações de qualidade fotográfica
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
                camera.set(cv2.CAP_PROP_FPS, 30)  # FPS alto para melhor qualidade
                
                # Configurações fotográficas avançadas
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposição para foto
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # Auto foco ativo
                camera.set(cv2.CAP_PROP_AUTO_WB, 1)           # Auto white balance
            
            # Verificar resolução final obtida
            final_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            final_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"📐 Photo resolution set to: {final_width}x{final_height}")
            
            # AGUARDAR ESTABILIZAÇÃO DA CÂMERA (importante para foto de qualidade)
            self.logger.info("⏳ Camera stabilization for photo quality...")
            time.sleep(3)  # Tempo maior para estabilização fotográfica
            
            # LIMPAR BUFFER (descartar frames de vídeo antigos)
            self.logger.info("🧹 Clearing video buffer for fresh photo...")
            frames_cleared = 0
            for i in range(10):
                ret, _ = camera.read()
                if ret:
                    frames_cleared += 1
                time.sleep(0.1)
            
            self.logger.info(f"🗑️ Cleared {frames_cleared} old video frames")
            
            # AGUARDAR PARA GARANTIR FRAME FOTOGRÁFICO FRESCO
            time.sleep(1.5)
            
            # TIRAR A FOTO FINAL
            self.logger.info("📸 CAPTURING PHOTO NOW...")
            
            ret, photo = camera.read()
            if not ret or photo is None or photo.size == 0:
                self.logger.error("❌ PHOTO CAPTURE FAILED")
                return None
            
            # Verificar qualidade da foto
            if photo.shape[0] < 200 or photo.shape[1] < 200:
                self.logger.error(f"❌ Photo resolution too low: {photo.shape}")
                return None
            
            # INFORMAÇÕES DA FOTO
            photo_width = photo.shape[1]
            photo_height = photo.shape[0]
            photo_channels = photo.shape[2] if len(photo.shape) > 2 else 1
            photo_size_mb = (photo.nbytes) / (1024 * 1024)
            
            current_time = datetime.datetime.now()
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Hash único da foto
            photo_hash = hashlib.md5(photo.tobytes()).hexdigest()[:12]  # Hash maior para fotos
            
            self.logger.info(f"✅ PHOTO CAPTURED SUCCESSFULLY!")
            self.logger.info(f"   📐 Resolution: {photo_width}x{photo_height}")
            self.logger.info(f"   💾 Size: {photo_size_mb:.2f} MB")
            self.logger.info(f"   🔢 Channels: {photo_channels}")
            self.logger.info(f"   🔑 Hash: {photo_hash}")
            self.logger.info(f"   🕐 Time: {timestamp_str}")
            
            # Verificar se não é foto duplicada
            if hasattr(self, 'last_photo_hash') and photo_hash == self.last_photo_hash:
                self.logger.warning(f"⚠️ DUPLICATE PHOTO detected! Hash: {photo_hash}")
                is_duplicate = True
            else:
                is_duplicate = False
                
            self.last_photo_hash = photo_hash
            
            # CODIFICAR FOTO EM MÁXIMA QUALIDADE
            self.logger.info("🎨 Encoding photo in high quality...")
            
            # Parâmetros para máxima qualidade JPEG
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 95,  # Qualidade muito alta (95%)
                cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Otimização ativa
            ]
            
            success, buffer = cv2.imencode('.jpg', photo, encode_params)
            if not success:
                self.logger.error("❌ Failed to encode photo")
                return None
            
            photo_base64 = base64.b64encode(buffer).decode('utf-8')
            encoded_size_mb = len(photo_base64) * 3/4 / (1024*1024)
            
            self.logger.info(f"📤 Photo encoded - Size: {encoded_size_mb:.2f} MB (base64)")
            
            # Obter propriedades técnicas da câmera
            camera_properties = {
                'fps': camera.get(cv2.CAP_PROP_FPS),
                'format': camera.get(cv2.CAP_PROP_FORMAT),
                'brightness': camera.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': camera.get(cv2.CAP_PROP_CONTRAST),
                'saturation': camera.get(cv2.CAP_PROP_SATURATION),
                'hue': camera.get(cv2.CAP_PROP_HUE),
                'gain': camera.get(cv2.CAP_PROP_GAIN),
                'exposure': camera.get(cv2.CAP_PROP_EXPOSURE),
                'auto_exposure': camera.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                'focus': camera.get(cv2.CAP_PROP_FOCUS) if hasattr(cv2, 'CAP_PROP_FOCUS') else None,
                'white_balance': camera.get(cv2.CAP_PROP_AUTO_WB) if hasattr(cv2, 'CAP_PROP_AUTO_WB') else None
            }
            
            photo_info = {
                'available': True,
                'width': photo_width,
                'height': photo_height,
                'channels': photo_channels,
                'raw_size_mb': photo_size_mb,
                'encoded_size_mb': encoded_size_mb,
                'timestamp': timestamp_str,
                'frame_hash': photo_hash,  # Manter compatibilidade
                'photo_hash': photo_hash,
                'is_duplicate': is_duplicate,
                'frames_cleared': frames_cleared,
                'photo_mode': True,
                'quality_level': 95,
                'frame': photo,  # Foto em formato numpy
                'frame_base64': photo_base64,  # Foto em base64
                **camera_properties
            }
            
            return photo_info
            
        except Exception as e:
            self.logger.error(f"❌ PHOTO CAPTURE ERROR: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
        finally:
            # SEMPRE FECHAR CÂMERA APÓS FOTO
            if camera is not None:
                self.logger.info("📷 CLOSING camera after photo...")
                camera.release()
                time.sleep(0.5)  # Aguardar liberação
                self.logger.info("✅ CAMERA CLOSED - Photo session ended")
                
    def start(self):
        """Inicia o sistema de monitoramento fotográfico."""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        if not self.model_available:
            self.logger.error("Model not available, cannot start system")
            return
        
        self.logger.info("📷 STARTING PHOTOGRAPHIC MONITORING SYSTEM")
        self.logger.info("   🎯 High-resolution photo snapshots")
        self.logger.info("   📷 Camera opens → takes photo → closes each cycle")
        
        # Limpar histórico de fotos
        if hasattr(self, 'last_photo_hash'):
            delattr(self, 'last_photo_hash')
        
        self.is_running = True
        
        # Iniciar thread de monitoramento fotográfico
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("✅ Photographic monitoring system started!")   
    def stop(self):
        """Para o sistema de monitoramento com cleanup adequado."""
        if not self.is_running:
            self.logger.warning("System is not running")
            return
        
        self.logger.info("Stopping monitoring system...")
        self.is_running = False
        
        # Limpar câmera IMEDIATAMENTE
        if hasattr(self, 'camera') and self.camera is not None:
            self.logger.info("Releasing camera...")
            self.camera.release()
            self.camera = None
            time.sleep(2)  # Aguardar liberação completa
        
        # Aguardar thread finalizar
        if hasattr(self, 'monitoring_thread'):
            self.logger.info("Waiting for monitoring thread to finish...")
            self.monitoring_thread.join(timeout=15)  # Timeout maior
            if self.monitoring_thread.is_alive():
                self.logger.warning("Monitoring thread did not finish gracefully")
        
        self.logger.info("Monitoring system stopped successfully")

    def test_photo_capture(self):
        """Testa captura de uma única foto"""
        self.logger.info("🧪 TESTING single photo capture...")
        
        photo_info = self._take_photo_snapshot()
        
        if photo_info:
            self.logger.info("✅ PHOTO TEST SUCCESSFUL!")
            self.logger.info(f"   📐 Resolution: {photo_info['width']}x{photo_info['height']}")
            self.logger.info(f"   💾 Size: {photo_info['raw_size_mb']:.2f} MB (raw)")
            self.logger.info(f"   📤 Size: {photo_info['encoded_size_mb']:.2f} MB (encoded)")
            self.logger.info(f"   🔑 Hash: {photo_info['photo_hash']}")
            self.logger.info(f"   ❓ Duplicate: {photo_info['is_duplicate']}")
            return True
        else:
            self.logger.error("❌ PHOTO TEST FAILED")
            return False
       

    def test_multiple_photos(self, count=3, interval=5):
        """Testa múltiplas fotos para verificar se são diferentes"""
        self.logger.info(f"🧪 TESTING {count} photos with {interval}s interval...")
        
        hashes = []
        sizes = []
        
        for i in range(count):
            self.logger.info(f"📷 Taking test photo {i+1}/{count}...")
            
            photo_info = self._take_photo_snapshot()
            
            if photo_info:
                hashes.append(photo_info['photo_hash'])
                sizes.append(photo_info['raw_size_mb'])
                self.logger.info(f"   ✅ Photo {i+1}: {photo_info['photo_hash']} ({photo_info['raw_size_mb']:.2f}MB)")
                
                if i < count - 1:  # Não fazer delay na última foto
                    self.logger.info(f"   ⏳ Waiting {interval}s...")
                    time.sleep(interval)
            else:
                self.logger.error(f"   ❌ Photo {i+1} failed")
                return False
        
        # Análise dos resultados
        unique_hashes = len(set(hashes))
        avg_size = sum(sizes) / len(sizes)
        
        self.logger.info(f"📊 TEST RESULTS:")
        self.logger.info(f"   📷 Photos taken: {count}")
        self.logger.info(f"   🔄 Unique photos: {unique_hashes}")
        self.logger.info(f"   📐 Average size: {avg_size:.2f} MB")
        
        if unique_hashes == count:
            self.logger.info("✅ ALL PHOTOS ARE UNIQUE - Camera working perfectly!")
            return True
        else:
            duplicates = count - unique_hashes
            self.logger.warning(f"⚠️ {duplicates} duplicate photos found")
            return False