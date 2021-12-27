import fire, os, json
import http.client, urllib.request, urllib.parse, urllib.error, base64
import classes.ClassOpenCV
import classes.ClassFaceAPI
import classes.ClassPerson
import classes.ClassPersonGroup
import classes.ClassConfig


class FacePI:
    def show_opencv(self):
        classes.ClassOpenCV.show_opencv("")

    def Signin(self):
        """
        刷臉簽到
        """
        #        imagepath = '202994853.jpg'
        #        imagepath = 'face4.jpg'
        #        self.detectLocalImage(imagepath)
        #
        # imageurl = 'https://cdn-news.readmoo.com/wp-content/uploads/2016/07/Albert_einstein_by_zuzahin-d5pcbug-1140x600.jpg'
        # imageurl = 'https://cdn2.momjunction.com/wp-content/uploads/2020/11/facts-about-albert-einstein-for-kids-720x810.jpg'
        # classes.ClassFaceAPI.Face().detectImageUrl(imageurl)
        imagepath = classes.ClassOpenCV.show_opencv()
        json_face_detect = classes.ClassFaceAPI.Face().detectLocalImage(imagepath)

    def Train(self, userData=None, personname=None):
        """1. 用 3 連拍訓練一個新人"""
        jpgimagepaths = []
        for i in range(3):
            jpgimagepath = classes.ClassOpenCV.show_opencv(hint=" (訓練第 " + str(i + 1) + " 張)")
            jpgimagepaths.append(jpgimagepath)

        if personname == None:
            personname = input("請輸入您的姓名: ")

        if userData == None:
            userData = input("請輸入您的說明文字(比如: 高師大附中國一仁): ")

        jpgtrainpaths = []
        for jpgimagepath in jpgimagepaths:
            filename = os.path.basename(jpgimagepath)
            home = os.path.expanduser("~")
            jpgtrainpath = os.path.join(
                home, "traindatas", userData, personname, filename
            )
            if not os.path.exists(os.path.dirname(jpgtrainpath)):
                os.makedirs(os.path.dirname(jpgtrainpath))
            os.rename(jpgimagepath, jpgtrainpath)
            jpgtrainpaths.append(jpgtrainpath)


        myconfig = classes.ClassConfig.Config()
        

        personAPI = classes.ClassPerson.Person()
        personAPI.add_personimages(myconfig['personGroupId'], personname, userData, jpgtrainpaths)
        personGroupapi = classes.ClassPersonGroup.PersonGroup()
        personGroupapi.train_personGroup(myconfig['personGroupId'])


if __name__ == "__main__":
    fire.Fire(FacePI)
