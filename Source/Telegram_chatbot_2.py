# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:52:49 2020

@author: zlibn
"""

import requests
class MTRobot:
    def __init__(self):
        """
        Explanation:
        -----------
        bot_token: MTRobot token
        bot_chatID: receiver's ID
        msg: message
        """
        
    def sendtext(bot_message):
        bot_token = '1311429851:AAEf5SdTWEb1TaOqYtWw4fo90TGHXnG2mOI'
        bot_chatID = '759131145'
        send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
        response = requests.get(send_text)
        
        return None
   
