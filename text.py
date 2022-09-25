import Bertsum_test, image_captioning


class NewsSumModule:
    def init(self):
        self.path = "경로" #weight(checkpoint)의 경로
    def load(self):
        self.model = Bertsum_test.get_model()
        self.model.load_cp(path)
        self.model.eval()
        #model에 weight 올리기
    def remove(self):
        del self.model
    def forward(self):
        self.load()
        self.sdt=Bertsum_test.txt2input()
        self.remove()
        return #list 3줄요약 해주는 code

class ImgCapModule: #api에서 이미지를 받아와서, 이미지캡셔닝 수행.
    def load(self,src):
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
        self.news_sum = News_sumModule()
        self.imgcap = imgcapModule()
        self.retrieval = RetrievalModule()

    def update(self):
        
        #기사요약
        return "ㅠㅁ해"
