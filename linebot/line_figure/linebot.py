import requests
import json

headers = {"Authorization":"Bearer 3Y+EW9huYnJ9h/1j56u9Zhn+Dfis7C9vxdTezx2tjP0JrJxNgz9uf6VcdHaTexkpgz6ZI8sZKPWgYgvzfThhdraoIiaNRp+RqsGo+wGRKo52qwaDKFdfdLvwAR9BTw4hL2q+26xucYG7OFL87uBxwgdB04t89/1O/w1cDnyilFU=","Content-Type":"application/json"}

body ={
  "size": {
    "width": 2500,
    "height": 843
  },
  "selected": "true",
  "name": "功能選單",
  "chatBarText": "操作表單",
  "areas": [
    {
      "bounds": {
        "x": 0,
        "y": 0,
        "width": 842,
        "height": 840
      },
      "action": {
        "type": "message",
        "text": "健康資訊"
      }
    },
    {
      "bounds": {
        "x": 842,
        "y": 0,
        "width": 809,
        "height": 843
      },
      "action": {
        "type": "message",
        "text": "送藥服務"
      }
    },
    {
      "bounds": {
        "x": 1640,
        "y": 0,
        "width": 860,
        "height": 843
      },
      "action": {
        "type": "message",
        "text": "巡邏功能"
      }
    }
    
  ]
}

'''
#新增
req = requests.request('POST', 'https://api.line.me/v2/bot/richmenu', 
                       headers=headers,data=json.dumps(body).encode('utf-8'))

print(req.text)
'''
'''
#圖片
from linebot import (
    LineBotApi, WebhookHandler
)
line_bot_api = LineBotApi('3Y+EW9huYnJ9h/1j56u9Zhn+Dfis7C9vxdTezx2tjP0JrJxNgz9uf6VcdHaTexkpgz6ZI8sZKPWgYgvzfThhdraoIiaNRp+RqsGo+wGRKo52qwaDKFdfdLvwAR9BTw4hL2q+26xucYG7OFL87uBxwgdB04t89/1O/w1cDnyilFU=')
with open("lab506.png", 'rb') as f:
    line_bot_api.set_rich_menu_image( "richmenu-d04d5455325f3507f44fac114b20e9da", "image/png", f)

'''
#啟用
req = requests.request('POST', 'https://api.line.me/v2/bot/user/all/richmenu/richmenu-d04d5455325f3507f44fac114b20e9da', 
                       headers=headers)

print(req.text)

