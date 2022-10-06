# -*- coding: utf-8 -*-
import requests
import json
import time

from tqdm import tqdm
import torch
import torchvision
import numpy as np
import Bertsum, image_captioning, Retrieval, db

import sys
import os
import pickle
import pymysql
import base64
import requests

class RetrievalModule:
    def __init__(self):
        f = open("/home/dhk1349/NBH/api_key")
        key = f.readline().replace("\n", "")
        self.api_key = key
        self.sum =[]
        print("Retrieval Module initialized")
        
    def load(self, date, date2):
        self.issue = Retrieval.issue_ranking(date, self.api_key)
        self.query = Retrieval.query_ranking(date, date2, self.api_key, 10)

    def remove(self):
        self.sum.clear()

    def forward(self, date, date2):
        self.load(date, date2)
        self.sum = self.issue + self.query
        news = Retrieval.news_info(self.api_key,self.sum)
        self.remove()
        return news


class NewsSumModule:
    def __init__(self, path):
        self.path = path
        self.args = Bertsum.get_args()
        print("News Summarization Module initialized")

    def load(self):
        self.model = Bertsum.get_model('-1')
        ckpt = torch.load(self.path, map_location=lambda storage, loc: storage)
        self.model.load_cp(ckpt)
        self.model.eval()

    def remove(self):
        del self.model

    def forward(self, src):
        result = []
        self.load()
        trainer = Bertsum.build_trainer(self.args, 0, self.model, None)
        for s in tqdm(src):
            s = s.replace("\n", "")
            s = s.split(". ")
            if '@' in  s[-1]:
                s = '. '.join(s[:-1])
            else:
                s = '. '.join(s)
            processed_text = Bertsum.txt2input(s)
            # print(f"process_text: {processed_text}")
            test_iter = Bertsum.make_loader(self.args, processed_text, 'cpu')
            out = trainer.summ(test_iter, 10000)
            # out = out.replace("\n", "")
            out = [list(filter(None, s.split('. ')))[i] for i in out[0][:3]]
            for idx in range(len(out)):
                out[idx] = out[idx].replace("\n", "")
                out[idx]+='.'
            result.append(out)
        self.remove()
        return result
    
class ImgCapModule: 
    def __init__(self, path, device="cpu"):
        self.device = device
        self.path = path
        self.totensor = torchvision.transforms.ToTensor()
        print("Image Caption Module initialized")

    def load(self):
        return image_captioning.get_model(self.path, False)

    def remove(self):
        del self.model, self.clip_model, self.preprocess, self.tokenizer

    def forward(self, urls):
        use_beam_search = False #@param {type:"boolean"}
        prefix_length = 10
        
        self.model, self.clip_model, self.preprocess, self.tokenizer = self.load()
        caption_list = []
        for url in urls:
            article_caption = []
            for u in url:
                img = image_captioning.get_img(u)
                img = self.preprocess(img).unsqueeze(0)
                img = img.to(self.device)
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
                article_caption.append(generated_text_prefix)
            caption_list.append(article_caption)
        self.remove()
        return caption_list



class UpdateModule:
    def __init__(self):
        self.news_sum = NewsSumModule("/home/dhk1349/NBH/model_step_100000.pt")
        self.imgcap = ImgCapModule("/home/dhk1349/NBH/clipcap.pt", "cpu")
        self.retrieval = RetrievalModule()
    
    def update_db(self):
        return

    def today(self):
        # interval in seconds
        retrieved = self.retrieval.forward('2022-09-30', '2022-10-01')
        # retrieved = retrieved[:5]
        print(f"{len(retrieved)} of news")
        content_list = []
        img_list = []
        for r in retrieved:
            content_list.append(r['content'])
            img_list.append(r['images'].split('\n'))

        summ = self.news_sum.forward(content_list)
        # print(summ)
        cap = self.imgcap.forward(img_list)
        # print(cap)
        # print(img_list)
        for idx, r in enumerate(retrieved):
            r['summ'] = summ[idx]
            r["caption"] = cap[idx]
        return retrieved 
        



if __name__=="__main__":
    print("Test Run")
    dl_server = UpdateModule()
    out = dl_server.today()
    print(out)
    with open("../db_data.pickle", 'rb') as f:
        db_data = pickle.load(f)
    conn, cursor = db.connect_RDS(db_data["host"], db_data["port"], db_data["username"], db_data["password"], db_data["database"])
    for new_obj in out:
        new_obj["summ"] = " ".join(new_obj["summ"])
        new_obj["caption"] = " ".join(new_obj["caption"])
        new_obj["published_at"] = new_obj["published_at"][:10]
        q = db.insert_news(new_obj["news_id"], new_obj["published_at"], new_obj["summ"], new_obj["caption"], issue_rank=0, keyword="None")
        print(q)
        cursor.execute(q)
        conn.commit()
    cursor.close()
    conn.close()
