#__________________//////////////////////_______________
#__For Openvino by Python written by M.Inoishi
#__person_vehicle_bike detection program
#__OpenVino2021R2
#__Ver3.0 2021/14/Apr
#_________________//////////////////////_______________

#Step0___ Pythonライブラリー、モジュールのインポート
import cv2										#OpenCVモジュール（画像、動画ファイルの処理）
import numpy as np								#Pythonでの数値計算を高速、効率化するモジュール
import logging as log							#処理中におけるLogの吐き出しモジュール
from time import time							#日付と時間を扱うモジュール
from operator import itemgetter					#リストなどから所望する要素を抽出するモジュール
from openvino.inference_engine import IECore	#OpenVinoを動作させるためのモジュール(IECore)

#Step1___ IECoreをieとしてインスタンス化　
ie = IECore()

#Step2___ person-vehicle-bike-detection-crossroad(物体　人、車、バイク）を使用するクラス
class road_detection:
	#本クラスの初期化メソッド
    #推論モデルの構築、使用するプラグインを指定して、プラグインへロード
    #推論モデルのInput条件も取得
	def __init__(self):
  		#推論モデルのパスを指定
		model = './ir_eng/person-vehicle-bike-detection-crossroad-1016'

		#read_network(OpenVino API)で実行可能なモデルを構築
		net = ie.read_network(model=model+'.xml', weights=model+'.bin')
		
		#使用するプラグインを指定		
		__plug="CPU"
		#__plug="GPU"
		#__plug="MYRIAD"
		#__plug="HETERO:FPGA,CPU"

		#使用する推論モデルのInput条件を取得
		print("Get_batch_size1=",net.batch_size)
		self.input_layer  = next(iter(net.input_info))	
		self.output_blobs = next(iter(net.outputs)) 
		self.model_n, self.model_c, self.model_h, self.model_w = net.input_info[self.input_layer].input_data.shape
		print("Input spec =",self.model_n,self.model_c,self.model_h,self.model_w)
  
		#load_network(opennVino API)にて推論モデルをプラグインにロード
		self.exec_net = ie.load_network(network=net, device_name=__plug)
		return

	#推論実行するメソッド
	#用意した動画データを、推論モデルのInput条件に加工
 	#person-vehicle-bike-detection-crossroad-1016推論モデルを使用して、推論実行
	def road_detection_infer(self,__image):
		self.cap_w = cap.get(3)
		self.cap_h = cap.get(4)
		in_frame = cv2.resize(__image, (self.model_w, self.model_h))
		in_frame = in_frame.transpose((2, 0, 1)) 
		in_frame = in_frame.reshape((self.model_n, self.model_c, self.model_h, self.model_w))

		#Python上で推論（インファー）同期実行するためのコマンド(OpenVino API) 
		#res = self.exec_net.infer(inputs={self.input_layer:in_frame}) 
  
		#Python上で推論（インファー）非同期実行するためのコマンド(OpenVino API)
		self.exec_net.requests[0].async_infer({self.input_layer: in_frame})
		request_status = self.exec_net.requests[0].wait(-1)
		res = self.exec_net.requests[0].outputs[self.output_blobs]
    
		return res	
  
	#推論結果をもとに、動画ファイル上に四角枠を表示するメソッド
	def road_detection_display(self,__res,__num,__image):
		for obj in __res[0][0]:
			if obj[2] > 0.08:
				__num = __num + 1
				xmin = int(obj[3] * self.cap_w)
				ymin = int(obj[4] * self.cap_h)
				xmax = int(obj[5] * self.cap_w)
				ymax = int(obj[6] * self.cap_h)
				#class_id = int(obj[1])
				color = (255, 0, 0)
				cv2.rectangle(__image, (xmin, ymin), (xmax, ymax), color, 2)
				cv2.imshow('Video',__image)
		return __num

	
	
#Step4___ main ___　メインプログラムを実行するためのスレッド（ここからプログラムの実行が指定される）
if __name__=='__main__':
	th1 = road_detection()            			#th1にroad_detectionをインスタンス化
	cap = cv2.VideoCapture('test1.mp4')      	#動画ファイルをオープン
              
 
	while cap.isOpened():						#入力がある限り実行
		__num = 0
		ret, __image = cap.read()
		key = cv2.waitKey(1)					#ESC入力でBrake
		if key == 27:
			brake

		__res 	= th1.road_detection_infer(__image)              	#推論実行                     
		__num 	= th1.road_detection_display(__res,__num,__image)   #推論結果を加工、表示           
		print('Count_____',__num)                                      
