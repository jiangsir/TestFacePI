import fire, os, json
import http.client, urllib.request, urllib.parse, urllib.error, base64
import classes.ClassFaceAPI
import classes.ClassOpenCV

class FacePI:


    def show_opencv(self):
        classes.ClassOpenCV.show_opencv('')

    def Signin(self):
        '''
        刷臉簽到
        '''
#        imagepath = '202994853.jpg'
#        imagepath = 'face4.jpg'
#        self.detectLocalImage(imagepath)
#        
        # imageurl = 'https://cdn-news.readmoo.com/wp-content/uploads/2016/07/Albert_einstein_by_zuzahin-d5pcbug-1140x600.jpg'
        # imageurl = 'https://cdn2.momjunction.com/wp-content/uploads/2020/11/facts-about-albert-einstein-for-kids-720x810.jpg'
        # classes.ClassFaceAPI.Face().detectImageUrl(imageurl)
        imagepath = classes.ClassOpenCV.show_opencv()
        classes.ClassFaceAPI.Face().detectLocalImage(imagepath)
        
if __name__ == '__main__':
    fire.Fire(FacePI)