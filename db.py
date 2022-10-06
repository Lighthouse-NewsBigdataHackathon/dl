import sys
import os

import pymysql
import base64
import requests
import logging

def connect_RDS(host, port, username, password, database):
    try:
        conn = pymysql.connect(host=host, user=username, passwd=password, db=database, port=port, use_unicode=True, charset='utf8')

        cursor = conn.cursor()
    except:
        logging.error('Failed to connect RDS')
        sys.exit(1)
    return conn, cursor


def insert_news(title, news_id, news_date, summary, caption, issue_rank=0, keyword="None"):
    q = f"INSERT INTO news_final (title, news_id, news_date, summary, caption, issue_rank, keyword) VALUES ('{title}', {news_id}, '{news_date}', '{summary}', '{caption}', {issue_rank}, '{keyword}');"
    return q
