import Bertsum_test, image_captioning


class NewsSumModule:
    def init(self):
        self.path = "경로" #weight(checkpoint)의 경로
    def load(self):
        self.model = Bertsum_test.get_model()
        #Bertsum_test 함수가져오기 함수를 가져오는데 
        # 어느정도까지 가져와야 하는가?
        #model에 weight 올리기
    def remove(self):
        del self.model
    def forward(self):
        self.load()
        self.txt2input()
        self.remove()
        return #list 3줄요약 해주는 code

class imgcapModule:
    def init(self):
        self.path = "경로" #weight의 경로
        
    def load(self,src):
        self.model = "jk"
        #image_captioning 함수가져오기 
        # 함수를 가져오는데 
        # 어느정도까지 가져와야 하는가?
        #model에 weight 올리기
    def remove(self):
        del model
    def forward(self):
        self.load()
        
        self.remove()
        return  


class UpdateModule:

    def init(self):
        self.news_sum = News_sumModule()
        self.imgcap = imgcapModule()
        self.retrieval = RetrievalModule()

    def update(self):
        
        #기사요약
        return "ㅠㅁ해"
