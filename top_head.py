#__________________//////////////////////_______________
#__For Openvino by Python written by M.Inoishi
#__Head pose detection program
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

#Step2___ head-pose-estimation-adas-0001(顔の向き）推論モデルを使用するクラス
class head_pose:
	#本クラスの初期化メソッド
    #推論モデルの構築、使用するプラグインを指定して、プラグインへロード
    #推論モデルのInput条件も取得
	def __init__(self):
		#推論モデルのパスを指定
		model = './ir_eng/head-pose-estimation-adas-0001'
  
  		#read_network(OpenVino API)で実行可能なモデルを構築
		net = ie.read_network(model=model+'.xml', weights=model+'.bin')
		
		#使用するプラグインをしてい			
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
		self.exec_net = ie.load_network(network=net, device_name=__plug,num_requests=1)
		return

	#推論実行するメソッド
	#用意したカメラのデータを、推論モデルのInput条件に加工
	#head-pose-estimation-adas-0001推論モデルを使用して、推論実行
	def head_pose_infer(self,frame):
		in_frame = cv2.resize(frame, (self.model_w, self.model_h))
		in_frame = in_frame.transpose((2, 0, 1)) 
		in_frame = in_frame.reshape((self.model_n, self.model_c, self.model_h, self.model_w))

		#Python上で推論（インファー）同期実行するためのコマンド(OpenVino API) 
		#res = self.exec_net.infer(inputs={self.input_layer:in_frame}) 
  
		#Python上で推論（インファー）非同期実行するためのコマンド(OpenVino API)
		self.exec_net.requests[0].async_infer({self.input_layer: in_frame})
		request_status = self.exec_net.requests[0].wait(-1)
		res = self.exec_net.requests[0].outputs

		return res

#Step3___ demoのクラス
class demo:
	#初期化メソッド
	def __init__(self):
		return

	#結果を表示するメソッド
	def head_demo_print(self,__res):
		res_y_fc = np.squeeze(__res['angle_y_fc'])
		res_p_fc = np.squeeze(__res['angle_p_fc'])
		res_r_fc = np.squeeze(__res['angle_r_fc'])
  
		print("yaw=",int(res_y_fc), "pitch=",int(res_p_fc),"roll=",int(res_r_fc))
		return
		
#Step4___ main ___　メインプログラムを実行するためのスレッド（ここからプログラムの実行が指定される）
if __name__=='__main__':
	th1 = head_pose()       					#th1にhead_poseをインスタンス化
	th2 = demo()                     			#th2にdemoをインスタンス化

	cap = cv2.VideoCapture(0)                  	#Defalt カメラを指定してオープン  

	while cap.isOpened():						#入力がある限り実行
		ret, __image = cap.read()
		cv2.imshow("Video",__image)
		key = cv2.waitKey(1)					#ESC入力でBrake
		if key == 27:
			brake

		__res	= th1.head_pose_infer(__image)	#推論実行
		th22 	= th2.head_demo_print(__res)	#推論結果を加工、表示
  
