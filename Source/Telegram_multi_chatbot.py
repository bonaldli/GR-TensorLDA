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
        
    def sendtext(worker_idx, bot_message):
        bot_token = ['1150140576:AAHYrC4x4uRaK2en5PwZdaQ2AVH6OCjHR2g', '1270567536:AAG9XChN-KBjY0znLlM6VTCF-Sh6b0zBY-U', '1304120516:AAF9lk7Z-uHAbvg1TMWk-ooSONguLjLCljo', '1098146192:AAHIgw7fwSk11_h0WCuC2i6ldvczHu7qKoY']
        bot_chatID = '759131145'
        send_text = 'https://api.telegram.org/bot' + bot_token[worker_idx] + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
        response = requests.get(send_text)
        
        return None
   
