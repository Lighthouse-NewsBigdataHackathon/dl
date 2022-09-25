import requests,json
issue_url = "http://tools.kinds.or.kr:8888/issue_ranking"#issue_ranking url
popular_url = "http://tools.kinds.or.kr:8888/query_rank"#query_rank url



payload_issue = {"access_key":"", #api key 넣는곳
       "argument": {
           "date": "2022-09-06",
            "provider": ["국민일보" #언론사는 or로 추가지정가능
            ]
        }
    }

res_issue = requests.post(url,data=json.dumps(payload_issue))
hong_issue= res_issue.json()
hongnews = hong_issue['return_object']['topics']
realhongnews = hongnews[0]['news_cluster']

#news_code로 text뽑기 - 3. 뉴스 상세 정보조회 api
payload_issuenews = {
    "access_key": "",   #api key 넣는곳
    "argument": {
        "news_ids": [
            "{0}".format(issue_news)
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
url = "http://tools.kinds.or.kr:8888/search/news"
res = requests.post(url, data=json.dumps(payload))
hong = res.json()
hongtext = (hong['return_object']['documents'])
jitext = (hongtext[0]['content'])
jitext = jitext.split(".")
realtext = ""
for i in range(0,len(jitext)):
    realtext = realtext + jitext[i] + ".\n\n"
print(realtext)




payload_popular = {
   "access_key": "",   #api key 넣는곳
    "argument": {
        "from": "2018-01-01",
        "until": "2018-01-01",
        "offset": 5,
        "target_access_key": ""
    }
}



res_issue = requests.post(url,data=json.dumps(payload))

hong_issue= res_issue.json()
hongimg = hong_issue['return_object']['topics']
b = hongimg[0]['news_cluster']
issue_news = b[5]




