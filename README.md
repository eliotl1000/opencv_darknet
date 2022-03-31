# opencv_darknet
想法:
結合「OpenCV移動偵測」與「YoloV4物件偵測」過濾監視器畫面的內容

「OpenCV移動偵測」優點：速度快，缺點：易受風吹草動，光影或其它環境因素的影響而出現假警報

「YoloV4物件偵測」缺點：大量計算，耗時，優點：偵測結果較不易受到環境干擾

結合上述兩種方法，可以兼具速度快與較好的過濾結果，免去人工長時間緊盯螢幕尋找特定畫面的麻煩。


作法：
利用「OpenCV移動偵測」簡單快速的特性，把大部份靜止不動的畫面過濾掉，剩下少數有動靜的畫面傳給「YoloV4物件偵測」進行檢查，若發現有人入鏡，自動存檔記錄下來。

呼叫「YoloV4物件偵測」的頻率被設計成動態的，從每5張畫面偵測1次，到每12張偵測1次，避免過多的假警報造成不斷偵測無用的畫面，拖累系統效能；呼叫「YoloV4物件偵測」的結果也回饋給「OpenCV移動偵測」，動態調整「OpenCV移動偵測」的靈敏度。

當發現畫面中有人出現時，「YoloV4物件偵測」的偵測頻率和「OpenCV移動偵測」的靈敏度會同時立即提高，以便詳細記錄該段時間的監視畫面。


參考代碼：
OpenCV 移動偵測:
https://www.learnpythonwithrune.org/opencv-and-python-simple-noise-tolerant-motion-detector/
Darknet YoloV4 物件偵測:
https://github.com/AlexeyAB/darknet
Darknet 官方提供的wieght權重檔，不適合用來辨識一般由下往下拍的監視器影像，需要蒐集監視器影像來重新訓練，才會有比較好的辨識結果。


安裝方式：

先在網上取得 AlexeyAB 的 darknet 專案程式代碼，放到個人電腦中，經過適當的編譯，再將本專案的程式放到 darknet 的根目錄中，即可執行 darknet_motion.py。
