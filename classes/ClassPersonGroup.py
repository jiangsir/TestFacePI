import http.client, urllib.request, urllib.parse, urllib.error, base64, json
import classes.ClassConfig
config = classes.ClassConfig.Config().readConfig()


class PersonGroup:
    def __init__(self):
        self.api_key = config["api_key"]
        self.host = config["host"]

    def train_personGroup(self):
        personGroupId=config["personGroupId"]
        print(
            "train_personGroup: 開始訓練一個 personGroup personGroupId=" + personGroupId + "。"
        )

        headers = {
            # Request headers
            "Ocp-Apim-Subscription-Key": self.api_key,
        }

        params = urllib.parse.urlencode({"personGroupId": personGroupId})

        try:
            conn = http.client.HTTPSConnection(self.host)
            conn.request(
                "POST",
                "/face/v1.0/persongroups/" + personGroupId + "/train?%s" % params,
                "{body}",
                headers,
            )
            response = conn.getresponse()
            data = response.read()
            print(data)
            conn.close()
        except Exception as e:
            print("[Errno {0}]連線失敗！請檢查網路設定。 {1}".format(e.errno, e.strerror))
