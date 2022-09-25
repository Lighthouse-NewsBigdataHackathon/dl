import requests
import json


import torch
import torchvision
import numpy as np
import Bertsum, image_captioning# ,  Retrieval


class NewsSumModule:
    def __init__(self, path):
        self.path = path
        self.args = Bertsum.get_args()

    def load(self):
        self.model = Bertsum.get_model('-1')
        ckpt = torch.load(self.path, map_location=lambda storage, loc: storage)
        self.model.load_cp(ckpt)
        self.model.eval()

    def remove(self):
        del self.model

    def forward(self, src):
        """
        scr: realtext를 list에 담아서 넣는다.
        """
        result = []
        self.load()
        trainer = Bertsum.build_trainer(self.args, 0, self.model, None)
        for s in src:
            processed_text = Bertsum.txt2input(realtext)
            test_iter = Bertsum.make_loader(self.args, processed_text, 'cpu')

            out = trainer.summ(test_iter, 10000)
            out = [list(filter(None, s.split('.')))[i] for i in out[0][:3]]
            result.append(out)
        self.remove()
        return result
    
class ImgCapModule: #api에서 이미지를 받아와서, 이미지캡셔닝 수행.
    def __init__(self, path, device="cpu"):
        self.device = device
        self.path = path
        self.totensor = torchvision.transforms.ToTensor()
    def load(self):
        return image_captioning.get_model(self.path, False)

    def remove(self):
        del self.model, self.clip_model, self.preprocess, self.tokenizer

    def forward(self, urls):
        use_beam_search = False #@param {type:"boolean"}
        prefix_length = 10
        
        self.model, self.clip_model, self.preprocess, self.tokenizer = self.load()
        # 빅카인즈 url 붙여서 img추출해야함
        # self.hong = image_captioning.img_cap()
        caption_list = []
        for url in urls:
            img = image_captioning.get_img(url)
            img = self.preprocess(img).unsqueeze(0).to(self.device)
            # img = self.totensor(img)
            with torch.no_grad():
                # if type(model) is ClipCaptionE2E:
                #     prefix_embed = model.forward_image(image)
                # else:
                prefix = self.clip_model.encode_image(img).to(self.device, dtype=torch.float32)
                prefix_embed = self.model.clip_project(prefix).reshape(1, prefix_length, -1)
            if use_beam_search:
                generated_text_prefix = image_captioning.generate_beam(self.model, self.tokenizer, embed=prefix_embed)[0]
            else:
                generated_text_prefix = image_captioning.generate2(self.model, self.tokenizer, embed=prefix_embed)
            caption_list.append(generated_text_prefix)
        self.remove()
        return caption_list



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
    print(result)
    """
    input_data = Bertsum.txt2input(realtext)
    sum_list = test(args, input_data, -1, '', None)
    sum_list[0]

    #Result
    out = [list(filter(None, realtext.split('.')))[i] for i in sum_list[0][0][:3]]
    print(out)
    """


    # image captioning test

    payload = {
        "access_key": " ",   #api key 넣는곳
        "argument": {
            "news_ids": [
                "02100601.20211027093629001"
            ],
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
    payload["access_key"]=key
    url = "http://tools.kinds.or.kr:8888/search/news"
    res = requests.post(url, data=json.dumps(payload))
    hong = res.json()
    hongimg = (hong['return_object']['documents'])                   #json형식의 return_object의 document list의 images를 가져오는 code
    realimg = (hongimg[0]['images'])
    superimg = "https://www.bigkinds.or.kr/resources/images"+realimg #앞의 url을 붙여야 이미지 획득 가능
    urls = [realimg, realimg]
    caption_module = ImgCapModule("/desktop/clipcap.pt", "cpu")
    captions = caption_module.forward(urls)
    print(captions)
