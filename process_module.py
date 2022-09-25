import torch
import numpy as np
import Bertsum, image_captioning, Retrieval


class NewsSumModule:
    def init(self, path):
        self.path = path
        self.args = Bertsum.get_args()

    def load(self):
        self.model = Bertsum.get_model()
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
            text = (s['return_object']['documents'])
            jitext = (text[0]['content'])                 #json 내의 return_object안의 documents list 내부의
            jitext = jitext.split(".")                        #content(기사)를 가져옴
            realtext = ""
            for i in range(0,len(jitext)):
                realtext = realtext + jitext[i]+"."     
            processed_text = Bertsum.txt2input(realtext)
            test_iter = Bertsum.make_loader(self.args, processed_text, "cuda")

            out = trainer.summ(test_iter, 10000)
            out = [list(filter(None, realtext.split('.')))[i] for i in out[0][:3]]
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

