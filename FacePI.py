import fire, os, json
import http.client, urllib.request, urllib.parse, urllib.error, base64

basepath = os.path.dirname(os.path.realpath(__file__))
configpath = os.path.join(basepath, 'Config.json')

class FacePI:
    def writeConfig(self, config):
        with open(configpath, 'w', encoding='utf-8') as f:
            json.dump(config, f)

    def readConfig(self):
        if not os.path.exists(configpath):
            config = dict()
            config['api_key'] = "b9160fbd882f47bd821205a4bce64354"
            config['host'] = "eastasia.api.cognitive.microsoft.com"
            config['confidence'] = 0.6
            config['title'] = '高師大附中多元選修'
            config['personGroupName'] = '預設人群名稱'
            config['personGroupId'] = 'default_personGroupId'
            self.writeConfig(config)

        with open(configpath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config


    def setConfig(self):
        config = self.readConfig()
        print('每個參數後的[]內代表目前的設定值，直接按 ENTER 代表不更改。')
        api_key = input(f'請輸入有效的 API_KEY[{config["api_key"]}]: ')
        if api_key: config['api_key'] = api_key
        title = input(f'請輸入 title[{config["title"]}]: ')
        if title: config['title'] = title

        self.writeConfig(config)
        #print(type(config))


    # 用本地端的圖檔進行辨識。
    def detectLocalImage(self, imagepath):
        headers = {
            # Request headers
            'Content-Type': 'application/octet-stream',  # 用本地圖檔辨識
            'Ocp-Apim-Subscription-Key': self.readConfig()['api_key'],
        }

        params = urllib.parse.urlencode({
            # Request parameters
            'returnFaceId':
            'true',
            'returnFaceLandmarks':
            'true',
            'returnFaceAttributes':
            'age,gender,emotion'
        })
        #'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure'
        print('imagepath=', imagepath)
        requestbody = open(imagepath, "rb").read()
        try:
            conn = http.client.HTTPSConnection(self.readConfig()['host'])
            conn.request("POST", "/face/v1.0/detect?%s" % params, requestbody,
                         headers)
            response = conn.getresponse()
            data = response.read()
            json_face_detect = json.loads(str(data, 'UTF-8'))
            print("detectLocalImage.faces=", json_face_detect)
            #print(parsed[0]['faceId'])
            #faceids.append(parsed[0]['faceId'])
            conn.close()


            print("detectLocalImage:",
                f"{imagepath} 偵測到 {len(json_face_detect)} 個人")

            return json_face_detect
            
        except Exception as e:
            print("[Errno {0}]連線失敗！請檢查網路設定。 {1}".format(e.errno, e.strerror))
            #return []

    def Indntify(self, imagepath):
        '''
        傳入圖片路徑, url 並進行辨識
        '''
        self.detectLocalImage(imagepath)



    def Signin(self):
        imagepath = '202994853.jpg'
        imagepath = 'face4.jpg'
        self.Indntify(imagepath)
        
if __name__ == '__main__':
    fire.Fire(FacePI)