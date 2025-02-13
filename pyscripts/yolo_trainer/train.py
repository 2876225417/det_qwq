

import json



def convert_json2config(json_path):
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    

    params = {
        'data':             cfg['basic']['data'],
        'epochs':           cfg['control']['epochs'],
        'patience':         cfg['control']['patience'],
        'batch':            cfg['control']['batch'],
        'imgsz':            cfg['control']['imgsz'],
        'save':             cfg['control']['save'],
        'saved_period':     cfg['control']['saved_period'],
        'cache':            cfg['basic']['cache'],
        'device':           cfg['device']['type'],
        'workers':          cfg['basic']['workers'],
        'project':          cfg['basic']['project'],
        'name':             cfg['basic']['name'],
        'exist_ok':         cfg['basic']['exist_ok'],
        'pretrained':       cfg['device']['pretrained'],
        'optimizer':        cfg['advanced']['optmizer'],
        'verbose':          cfg['advanced']['verbose'],
        'seed':             cfg['advanced']['seed'],
        'deterministic':    cfg['advanced']['deterministic'],
        'single_cls':       cfg['advanced']['single_cls'],
        'rect':             cfg['advanced']['rect'],
        'cos_lr':           cfg['advanced']['cos_lr'],
        'resume':           cfg['advanced']['resume'],
        'amp':              cfg['optimizer']['amp']
    }

    float_params = [
        ('lr0',             cfg['optimizer']['lr0'], 4),
        ('lrf',             cfg['optimizer']['lrf'], 4),
        ('momentum',        cfg['optimizer']['momentum'], 3),
        ('weight_decay',    cfg['optimizer']['weight_decay'], 6),
        ('warmup_epochs',   cfg['advanced']['warmup_epochs'], 1),
        ('warmup_momentum', cfg['advanced']['warmup_momentum'], 2),
        ('warmup_bias_lr',  cfg['advanced']['warmup_bias_lr'], 2)
    ]

    for key, value, precision in float_params:
        params[key] = round(value, precision)
    
    aug_map = {
        'hsv_h': cfg['augmentation']['hsv_h'],
        'hsv_s': cfg['augmentation']['hsv_s'],
        'hsv_v': cfg['augmentation']['hsv_v'],
        'degrees': cfg['augmentation']['degrees'],
        'translate': cfg['augmentation']['translate'],
        'scale': cfg['augmentation']['scale'],
        'shear': cfg['augmentation']['shear'],
        'perspective': cfg['augmentation']['perspective'],
        'flipud': cfg['augmentation']['flipud'],
        'fliplr': cfg['augmentation']['fliplr'],
        'mosaic': cfg['augmentation']['mosaic'],
        'mixup': cfg['augmentation']['mixup'],
        'copy_paster': cfg['augmentation']['copy_paste']
    }
    params.update(aug_map)

    params.update({
        'box': cfg['loss']['box'],
        'cls': cfg['loss']['cls'],
        'dfl': cfg['loss']['dfl'],
        'pose': cfg['loss']['pose'],
        'kobj': cfg['loss']['kobj'],
        'label_smoothing': cfg['loss']['label_smoothing']   
    })


    device_map = {
        
    }

from websockets.sync import server
import threading
from ultralytics import YOLO
from ultralytics import RTDETR
import uuid
from datetime import datetime

class WebSocketServer:
    def __init__(self, port=8765):
        self.port = port
        self.websocket = None
        self.server_thread = threading.Thread(target=self._start_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def _start_server(self):
        with server.serve(self._handler, "0.0.0.0", self.port) as ws_server:
            print(f"WebSocket server started on port {self.port}")
            ws_server.serve_forever()

    def _handler(self, websocket):
        self.websocket = websocket
        while True:
            try:
                websocket.recv()  # 保持连接
            except:
                break

    def send(self, data):
        if self.websocket:
            self.websocket.send(str(data))
              
# ================= 训练监控类 =================
class TrainingMonitor:
    def __init__(self, ws_port=8765):
        # 初始化WebSocket
        self.ws_server = WebSocketServer(ws_port)
        # 训练元数据
        self.train_id = str(uuid.uuid4())[:8]  # 生成8位训练ID
        self.start_time = datetime.now().isoformat()
        # 初始化日志文件
        self.log_file = open('training_metrics.csv', 'w')
        self.log_file.write('train_id,epoch,batch,loss,lr,event_type,timestamp\n')
        self.total_epochs = 0
    def on_train_start(self, trainer):
        """训练开始时触发"""
        try:
            total_batches = len(trainer.train_loader.dataset)
            self.total_epochs = getattr(trainer.args, 'epochs', 100)
            data = {
                'event_type': 'train_start',
                'train_id': self.train_id,
                'start_time': self.start_time,
                'total_batches': total_batches,
                'timestamp': datetime.now().isoformat(),
                'total_epochs': self.total_epochs
            }
            
            self._log_to_csv({
                'train_id': self.train_id,
                'epoch': 0,
                'batch': total_batches,
                'event_type': 'train_start',
                'timestamp': data['timestamp']
            })
            
            self.ws_server.send(json.dumps(data))
            print(f"✅ Training {self.train_id} started | Total batches: {total_batches}")
            
        except Exception as e:
            print(f"启动监控异常: {str(e)}")

    def on_train_epoch_end(self, trainer):
        self.total_epochs = getattr(trainer.args, 'epochs', 100)
        try:
            metrics = {
                'train_id': self.train_id,
                'epoch': trainer.epoch + 1,
                "batch": len(trainer.train_loader.dataset),
                'loss': trainer.loss.item(),  
                'lr': trainer.optimizer.param_groups[0]['lr'],
                'event_type': 'epoch_end',
                'timestamp': datetime.now().isoformat(),
                'total_epochs': self.total_epochs
            }
            
            self._log_to_csv(metrics)
            self.ws_server.send(json.dumps(metrics))
            
            print(f"Epoch {metrics['epoch']} | Batch: {metrics['batch']} | Loss: {metrics['loss']:.4f} | Lr: {metrics['lr']:.4f}")

        except Exception as e:
            print(f"Epoch监控异常: {str(e)}")

    def on_train_end(self, trainer):
        """训练结束时触发"""
        try:
            end_time = datetime.now().isoformat()
            model = trainer.model
            model_info = {
                'params': sum(p.numel() for p in model.parameters()), # 参数量
                'layers': len(model.model),  # 总层数
                'gradients': sum(p.requires_grad for p in model.parameters()), 
                'gflops': getattr(model, 'gf', 0)  # 需要根据实际模型结构获取
            }
            # 收集验证指标
            val_metrics = {
                'mAP50': trainer.metrics.get('metrics/mAP50(B)', 0),
                'mAP5095': trainer.metrics.get('metrics/mAP50-95(B)', 0),
                'precision': trainer.metrics.get('metrics/precision(B)', 0),
                'recall': trainer.metrics.get('metrics/recall(B)', 0)
            }

            duration = datetime.now() - datetime.fromisoformat(self.start_time)

            report = {
                'event_type': 'train_end',
                'train_id': self.train_id,
                'model_info': model_info,
                'start_time': self.start_time,
                'speed_stats': {
                    'preprocess': trainer.metrics.get('Speed/preprocess(ms)', 0),
                    'inference': trainer.metrics.get('Speed/inference(ms)', 0),
                    'postprocess': trainer.metrics.get('Speed/postprocess(ms)', 0)
                },
                'val_metrics': val_metrics,
                'save_path': str(trainer.save_dir),
                'duration': str(duration),
                'end_time': end_time,
                'remaining_batches': 0,  # 实际应根据实现补充剩余批次逻辑
                'timestamp': end_time
            }
            
            self._log_to_csv({
                'train_id': self.train_id,
                'epoch': trainer.epoch + 1,
                'batch': 0,
                'event_type': 'train_end',
                'timestamp': end_time
            })
            
            self.ws_server.send(json.dumps(report))
            print(f"⏹ Training {self.train_id} finished | Duration: {end_time}")
            
        except Exception as e:
            print(f"结束监控异常: {str(e)}")
        finally:
            self.log_file.close()

    def _log_to_csv(self, data):
        """统一日志记录方法"""
        line = (
            f"{data.get('train_id', '')},"
            f"{data.get('epoch', 0)},"
            f"{data.get('batch', 0)},"
            f"{data.get('loss', 0):.4f},"
            f"{data.get('lr', 0):.6f},"
            f"{data.get('event_type', 'unknown')},"
            f"{data.get('timestamp', '')}\n"
        )
        self.log_file.write(line)
        self.log_file.flush()

    def __del__(self):
        if hasattr(self, 'ws_server') and self.ws_server.websocket:
            self.ws_server.websocket.close()

if __name__ == '__main__':
    # 加载模型
    # model = YOLO(r'cfg/models/11/yolo11.yaml')  # 不使用预训练权重训练
    model = YOLO(r'cfg/models/11/yolo11.yaml').load("yolo11m.pt")  # 使用预训练权重训练
    monitor = TrainingMonitor()

        # 注册三个关键回调
    model.add_callback("on_train_start", monitor.on_train_start)
    model.add_callback("on_train_epoch_end", monitor.on_train_epoch_end)
    model.add_callback("on_train_end", monitor.on_train_end)
 
     # 训练参数 ----------------------------------------------------------------------------------------------
    model.train(
        data=r'HA.yaml',
        epochs=5,  # (int) 训练的周期数
        patience=50,  # (int) 等待无明显改善以进行早期停止的周tiao期数
        batch=32,  # (int) 每批次的图像数量（-1 为自动批处理）
        imgsz=640,  # (int) 输入图像的大小，整数或w，h
        save=True,  # (bool) 保存训练检查点和预测结果
        save_period=-1,  # (int) 每x周期保存检查点（如果小于1则禁用）
        cache=False,  # (bool) True/ram、磁盘或False。使用缓存加载数据
        device='0',  # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=4,  # (int) 数据加载的工作线程数（每个DDP进程）
        project='runs/train',  # (str, optional) 项目名称
        name='exp',  # (str, optional) 实验名称，结果保存在'project/name'目录下
        exist_ok=False,  # (bool) 是否覆盖现有实验
        pretrained=True,  # (bool | str) 是否使用预训练模型（bool），或从中加载权重的模型（str）
        optimizer='SGD',  # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
        verbose=True,  # (bool) 是否打印详细输出
        seed=0,  # (int) 用于可重复性的随机种子
        deterministic=True,  # (bool) 是否启用确定性模式
        single_cls=False,  # (bool) 将多类数据训练为单类
        rect=False,  # (bool) 如果mode='train'，则进行矩形训练，如果mode='val'，则进行矩形验证
        cos_lr=False,  # (bool) 使用余弦学习率调度器
        close_mosaic=0,  # (int) 在最后几个周期禁用马赛克增强
        resume=False,  # (bool) 从上一个检查点恢复训练
        amp=True,  # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
        fraction=1.0,  # (float) 要训练的数据集分数（默认为1.0，训练集中的所有图像）
        profile=False,  # (bool) 在训练期间为记录器启用ONNX和TensorRT速度
        freeze=None,  # (int | list, 可选) 在训练期间冻结前 n 层，或冻结层索引列表。
        # 分割
        overlap_mask=True,  # (bool) 训练期间是否应重叠掩码（仅适用于分割训练）
        mask_ratio=4,  # (int) 掩码降采样比例（仅适用于分割训练）
        # 分类
        dropout=0.0,  # (float) 使用丢弃正则化（仅适用于分类训练）
        # 超参数 ----------------------------------------------------------------------------------------------
        lr0=0.01,  # (float) 初始学习率（例如，SGD=1E-2，Adam=1E-3）
        lrf=0.01,  # (float) 最终学习率（lr0 * lrf）
        momentum=0.937,  # (float) SGD动量/Adam beta1
        weight_decay=0.0005,  # (float) 优化器权重衰减 5e-4
        warmup_epochs=3.0,  # (float) 预热周期（分数可用）
        warmup_momentum=0.8,  # (float) 预热初始动量
        warmup_bias_lr=0.1,  # (float) 预热初始偏置学习率
        box=7.5,  # (float) 盒损失增益
        cls=0.5,  # (float) 类别损失增益（与像素比例）
        dfl=1.5,  # (float) dfl损失增益
        pose=12.0,  # (float) 姿势损失增益
        kobj=1.0,  # (float) 关键点对象损失增益
        label_smoothing=0.0,  # (float) 标签平滑（分数）
        nbs=64,  # (int) 名义批量大小
        hsv_h=0.015,  # (float) 图像HSV-Hue增强（分数）
        hsv_s=0.7,  # (float) 图像HSV-Saturation增强（分数）
        hsv_v=0.4,  # (float) 图像HSV-Value增强（分数）
        degrees=0.0,  # (float) 图像旋转（+/- deg）
        translate=0.1,  # (float) 图像平移（+/- 分数）
        scale=0.5,  # (float) 图像缩放（+/- 增益）
        shear=0.0,  # (float) 图像剪切（+/- deg）
        perspective=0.0,  # (float) 图像透视（+/- 分数），范围为0-0.001
        flipud=0.0,  # (float) 图像上下翻转（概率）
        fliplr=0.5,  # (float) 图像左右翻转（概率）
        mosaic=1.0,  # (float) 图像马赛克（概率）
        mixup=0.0,  # (float) 图像混合（概率）
        copy_paste=0.0,  # (float) 分割复制-粘贴（概率）
    )

