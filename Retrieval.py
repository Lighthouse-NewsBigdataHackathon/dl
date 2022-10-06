import requests,json
issue_url = "http://tools.kinds.or.kr:8888/issue_ranking"#issue_ranking url
popular_url = "http://tools.kinds.or.kr:8888/query_rank"#query_rank url
url = "http://tools.kinds.or.kr:8888/search/news"


# 4. issue_ranking api사용
def issue_ranking(date,api_key):
    payload_issue = {"access_key":api_key, #api key 넣는곳
        "argument": {
            "date": date,
                "provider": [
                ]
            }
        }

    res_issue = requests.post(issue_url,data=json.dumps(payload_issue))
    issue_json= res_issue.json()
    hong_issue = issue_json['return_object']['topics']
    newscluster_issue = []
    for i in range(len(hong_issue)):
        # for k in range(len(hong_issue[i]['news_cluster'])):
        newscluster_issue.append(hong_issue[i]['news_cluster'][0])   #newscluster_issuelist 안에 newscode들 저장되어있음
    print(len(newscluster_issue))
    return newscluster_issue

# 7. query_rank api사용
def query_ranking(date1,date2,api_key,offset):
    payload_popular = {
    "access_key": api_key,   #api key 넣는곳
        "argument": {
            "from": date1,
            "until": date2,
            "offset": offset,
            "target_access_key": ""
        }
    }
    res_popular = requests.post(popular_url,data=json.dumps(payload_popular))
    popular_json= res_popular.json()
    hong_popular = popular_json['return_object']['queries']
    #queries list의 query를 or로 넣어야함.
    querylist = ""
    for i in range(len(hong_popular)-1):
        querylist+=hong_popular[i]['query']
        querylist+=" OR "
    querylist+=hong_popular[-1]['query']

#query로 text뽑기 - 2.뉴스검색 api (3의 상세뉴스조회와 같은출력값)

    payload_popularnews = {
        "access_key": api_key,    #api key 넣는곳
        "argument": {
            "query": "{0}".format(querylist),
            "published_at": {
                "from": date1,
                "until": date2
            },
            "provider": [
            ],
            "category": [
            ],
            "category_incident": [
            ],
            "byline": "",
            "provider_subject": [
            ],
            "subject_info": [
                ""
            ],
            "subject_info1": [
                ""
            ],
            "subject_info2": [
                ""
            ],
            "subject_info3": [
                ""
            ],
            "subject_info4": [
                ""
            ],
            "sort": {"date": "desc"},
            "hilight": 200,
            "return_from": 0,
            "return_size": 5,
            "fields": [
                "byline",
                "category",
                "category_incident",
                "provider_news_id"
            ]
        }
    }

    popularnews_res = requests.post(url, data=json.dumps(payload_popularnews))
    popularnews = popularnews_res.json()
    hong_popularnews = (popularnews['return_object']['documents'])
    text_popularnews = []
    for i in range(len(hong_popularnews)):
        text_popularnews.append(hong_popularnews[i]['news_id'])
    print(len(text_popularnews))
    return text_popularnews

#newscluster_issue list안에는 issue_ranking(4)에서 뽑은 news_id들이,
#text_popularnews list안에는 query_rank(7) to news검색 api(2)에서 뽑은 news_id들이 들어가 있습니다.


#news_cluster(code)로 기사뽑기 - 3. 뉴스 상세 정보조회 api
def news_info(api_key,news_id):
    payload_issuenews = {
        "access_key": api_key,   #api key 넣는곳
        "argument": {
            "news_ids": news_id,
            "fields": [
                "title",
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

    issuenews_res = requests.post(url, data=json.dumps(payload_issuenews))
    issuenews = issuenews_res.json()
    hong_documents = (issuenews['return_object']['documents'])
    meta_container=[]
    text_issuenews=[]
    image_issuenews=[]
    keys = ['news_id','content','published_at', 'provider', 'images', 'category', 'title']
    for i in range(len(hong_documents)):
        meta_dict = {}
        for k in range(len(keys)):
            if keys[k]=="content":
                meta_dict[keys[k]] = (hong_documents[i][keys[k]].replace("\n", ""))
            else:
                meta_dict[keys[k]] = (hong_documents[i][keys[k]])
        meta_container.append(meta_dict)
    return meta_container
