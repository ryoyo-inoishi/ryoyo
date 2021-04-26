#__________________//////////////////////_______________
#__For Openvino by Python written by M.Inoishi
#__Squeezenet1.1 program
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
ie = IECore()									#インファレンスエンジンをieとしてインスタンス化

#Step2___ Squeezenet1.1を使用するためのクラス
class squeesenet:
    #本クラスの初期化メソッド
    #推論モデルの構築、使用するプラグインを指定して、プラグインへロードし
    #推論モデルのInput条件も取得
	def __init__(self,__image): 						
		#推論モデルのパスを指定
		model = './ir_eng/squeezenet1.1'									
		self.image = __image
  
		#read_network(OpenVino API)で実行可能なモデルを構築
		net = ie.read_network(model=model+'.xml', weights=model+'.bin')	
		#使用するプラグインを指定
		__plug ='CPU'														
		#__plug='GPU'
		#__plug='MYRIAD'
		#__plug='HETERO:FPGA,CPU'
  
		#使用する推論モデルのInput条件を取得
		print("Get_batch_size1=",net.batch_size)
		self.input_layer = next(iter(net.input_info))
		self.output_blobs = next(iter(net.outputs)) 
		self.model_n, self.model_c, self.model_h, self.model_w = net.input_info[self.input_layer].input_data.shape
		print("Input Spec = ",self.model_n,self.model_c,self.model_h,self.model_w)
  
		#load_network(opennVino API)にて推論モデルをプラグインにロード
		self.exec_net = ie.load_network(network=net, device_name=__plug)		
		return

	#推論実行するメソッド
	#用意した画像データを、推論モデルのInput条件に加工
 	#Squeezenetの推論モデルを使用して、推論実行
	def squeeze_infer(self):						
		#読み込んだイメージデータを加工します。cv2ライブラリを使用
		img_face = cv2.imread(self.image)									
		in_frame = cv2.resize(img_face, (self.model_w,self.model_h))	
		in_frame = cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB)		
		in_frame = in_frame.transpose((2, 0, 1))
		cv2.imshow('Image',img_face)					
		cv2.waitKey(2000)							
		cv2.destroyAllWindows()					

		#Python上で推論（インファー）同期実行するためのコマンド(OpenVino API) 
		#res = self.exec_net.infer(inputs={self.input_layer:in_frame}) 
  
		#Python上で推論（インファー）非同期実行するためのコマンド(OpenVino API)
		self.exec_net.requests[0].async_infer({self.input_layer: in_frame})
		request_status = self.exec_net.requests[0].wait(-1)
		res = self.exec_net.requests[0].outputs[self.output_blobs]

		return res															

#Step3___ squeezenetで推論した結果(res)を、見やすく加工するクラス
class squeeze_label:
    #本クラスの初期化メソッド
	def __init__(self,__label,__res):									
		self.__label = __label
		self.__res = __res
		self.values=[]															
		self.list=[]
  
		return

	#推論結果(0-999)をリスト化。精度の高い順に並び変え。Top５を表示するメソッド
	def squeeze_label(self):	
		#item変数に推論結果"prob"次元0のデータを追加
		#for item in self.__res['prob'][0]:										
		for item in self.__res[0]:										
		    self.values.append(item[0][0])						
		
		#lines変数に読み込んだラベルファイルの中身を追加
		with open(self.__label) as file:								
		    lines = file.readlines() 											

		#item変数に0-999
		for item in range(0,999): 												
			self.list.append(["{:.5f}".format(self.values[item]*1),item,lines[item].splitlines()])

		#表示する結果（リスト）を上位５までを表示
		print("__sort list__")	
		self.list.sort(key=itemgetter(0),reverse=True)	
		for cnt in range(0,5):													
			print(self.list[cnt])		
		print("__End__")
		return																

#Step4___ main ___　メインプログラムを実行saするためのスレッド（ここからプログラムの実行が指定される）
if __name__=='__main__':
	__label = "squeezenet1.1.labels"			#Squeezenetで定義されているラベルファイルを読み込み
	
#_______Run 1_________#
	__image ="cup.png"							#画像データを用意
	th1 	= squeesenet(__image)				#th1にクラスsqueezeをインスタンス化。変数として画像データ(__image)を渡す
	__res	= th1.squeeze_infer()  				#__resにth1.squeeze_inferメソッドを実行。推論結果(戻り値)を受け取る						
	th2 	= squeeze_label(__label,__res)		#th2にクラスsqueeze_labelをインスタンス化。変数としてラベルデータと、推論結果を渡す
	th21 	= th2.squeeze_label()				#th21にth2.squeeze_labelメソッドをインスタンス化。推論結果Top5を表示して終了。戻り値なし		

	print("___________________________")

#_______Run 2_________#
	__image ="car.png"							#画像データを用意
	th3 	= squeesenet(__image)            	#th3にクラスsqueezeをインスタンス化。変数として画像データ(__image)を渡す						
	__res 	= th3.squeeze_infer()				#__resにth3.squeeze_inferメソッドを実行。推論結果(戻り値)を受け取る
	th4 	= squeeze_label(__label,__res)    	#th4にクラスsqueeze_labelをインスタンス化。変数としてラベルデータと、推論結果を渡す  		
	th41 	= th4.squeeze_label()				#th41にth4.squeeze_labelメソッドをインスタンス化。推論結果Top5を表示して終了。戻り値なし
	print("___________________________")
