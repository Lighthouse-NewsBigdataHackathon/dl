import requests
import json


import torch
import numpy as np
import Bertsum# , image_captioning, Retrieval


class NewsSumModule:
    def __init__(self, path):
        self.path = path
        self.args = Bertsum.get_args()

    def load(self):
        self.model = Bertsum.get_model('0')
        self.model.to("cuda")
        ckpt = torch.load(self.path, map_location=lambda storage, loc: storage)
        self.model.load_cp(ckpt)
        self.model.eval()

    def remove(self):
        del self.model

    def forward(self, src):
        """
        scr: list of inputs
        """
        result = []
        self.load()
        trainer = Bertsum.build_trainer(self.args, 0, self.model, None)
        for s in src:
            processed_text = Bertsum.txt2input(realtext)
            test_iter = Bertsum.make_loader(self.args, processed_text, 0)

            out = trainer.summ(test_iter, 10000)
            out = [list(filter(None, s.split('.')))[i] for i in out[0][:3]]
            result.append(out)
        self.remove()
        return result
    
class ImgCapModule: #api에서 이미지를 받아와서, 이미지캡셔닝 수행.
    def load(self):
        self.model =  "retrieval에서 이미지 받아오기."
    def remove(self):
        del model
    def forward(self):
        self.load()
        #빅카인즈 url 붙여서 img추출해야함

        self.hong = image_captioning.img_cap()
        self.remove()
        return self.hong



class UpdateModule:

    def init(self):
        self.news_sum = NewsSumModule()
        self.imgcap = imgcapModule()
        self.retrieval = RetrievalModule()

    def update(self):
        
        #기사요약
        return "ㅠㅁ해"

if __name__=="__main__":
    print(torch.cuda.device_count())
    news_sum = NewsSumModule("/desktop/model_step_100000.pt")
    #API에서 가져온 text를 집어넣는 CODE
    f = open("/desktop/api_key")
    key = f.readline().replace("\n", "")
    #issue_ranking code : news code 추출용 
    payload_issue = {"access_key":"",   # api key 넣는곳 
        "argument": {
            "date": "2022-09-13",
            "provider": ["국민일보"
            ]
        }
    }
    payload_issue["access_key"]=key
    # print(payload_issue)
    url_issue = "http://tools.kinds.or.kr:8888/issue_ranking"
    res_issue = requests.post(url_issue,data=json.dumps(payload_issue))
    # print(res_issue.content)
    # hong_issue= res_issue.json()   #json파일로 받은 data
    hong_issue = json.loads(res_issue.content, encoding='utf-8', strict=False)
    hongimg = hong_issue['return_object']['topics']   
    b = hongimg[0]['news_cluster']                    #b는 return_object의 topics의 list내의 news_cluster list
    issue_news = b[4]                                 #issue_news라는 변수내에 b[4]를 넣고 test
                                                    #어떤 방식으로 쓰일지 몰라 우선은 가시적으로 확인하기 위해 정수를 넣어 코드작성함
    #news_code로 기사 가져오기
    payload = {
        "access_key": "",          #api key 넣는곳
        "argument": {
        "news_ids": b,     #issue_news 변수로 news_id가져오기
        "fields": [
            "content",
            "byline",
            "category",
            "category_incident",
            "images",
            "provider_subject",
            "provider_news_id",
            "publisher_code"
        ]
        }
    }
    payload["access_key"] = key
    url = "http://tools.kinds.or.kr:8888/search/news"
    res = requests.post(url, data=json.dumps(payload))
    hong = res.json()
    hongtext = (hong['return_object']['documents'])
    print(f"total {len(hongtext)} articles")
    input_list = []
    for article in range(len(hongtext)):
        jitext = (hongtext[article]['content'])                 #json 내의 return_object안의 documents list 내부의
        jitext = jitext.split(".")                        #content(기사)를 가져옴
        realtext = ""
        for i in range(0,len(jitext)):
            realtext = realtext + jitext[i]+"."          #마침표를 기준으로 기사를 split하여 다시 마침표를 붙여 넣어줌
        input_list.append(realtext)
        # print(realtext)
    print(input_list)
    result = news_sum.forward(input_list)

    input_data = Bertsum.txt2input(realtext)
    sum_list = test(args, input_data, -1, '', None)
    sum_list[0]

    #Result
    out = [list(filter(None, realtext.split('.')))[i] for i in sum_list[0][0][:3]]
    print(out)
