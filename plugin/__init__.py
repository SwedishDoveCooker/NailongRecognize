# import re
# import mimetypes
# import aiohttp
import httpx
import filetype
# from urllib.parse import urlparse
# from random import choice
from pathlib import Path
# from typing import Optional
from nonebot.rule import is_type
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata
from nonebot.permission import SUPERUSER
from nonebot import logger
from nonebot import on_command
from nonebot import get_bot
from nonebot.plugin import on_message
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    Message,
    MessageSegment
)
from nonebot.adapters.onebot.v11 import PrivateMessageEvent, GroupMessageEvent
from nonebot.log import default_format, default_filter
from random import randint
# from .tf_idf import compute_idf, rank_documents
# from .simple_search import simple_search, very_ex_name_handler, name_handler
# from .image_r3cognition import rename_images, process_images
from .img import *
import hashlib,uuid
# logger.add("info.log", level="DEBUG", format=default_format, rotation="10 days", compression="zip")
__dir = Path(__file__).parent

seer = on_message()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")
model_path = __dir.joinpath("nailong.pth")
model = load_model(model_path, device)

@seer.handle()
async def handle_message_seer(bot: Bot, event: Event):
    name: str = event.get_plaintext()
    flag: list = []
    unique_folder_name:str = str(uuid.uuid4())
    for i in event.get_message():
        # logger.info(i.__dict__)
        if i.__dict__['type'] == 'image':
            # logger.info("fetched "+i.__dict__['data']['url'])
            flag.append(i.__dict__['data']['url'])
    # logger.info(event.get_message())
    # logger.info(event.get_message()[0].__dict__)
    # logger.info(event.get_message()[0].__dict__['data']['url'])
    # logger.info(event.get_message()[1].__dict__)
    # logger.info(event.get_message()[1].__dict__['data']['url'])
    if flag:
        for URL in flag:
            '''
            async with aiohttp.ClientSession() as session:
                r, extension = await fetch(session, URL)
                if r:
                    md5_hash = hashlib.md5(r).hexdigest()
                    full_filename = f"{md5_hash}{extension}"
                    folder_path = __dir.joinpath("assets", "uploads", full_filename)
                    with open(folder_path, 'wb') as f:
                        f.write(r)
                    logger.info(f"Image saved: {full_filename} from URL: {URL}")
            '''
            async with httpx.AsyncClient() as client:
                response = await client.get(URL)
                if response.status_code == 200:
                    md5_hash = hashlib.md5(response.content).hexdigest()
                    kind = filetype.guess(response.content)
                    filename = f"{md5_hash}.{kind.extension}"
                    save_path = __dir.joinpath("input", unique_folder_name, filename)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    # logger.info(f"Image saved: {filename} from URL: {URL}")
                else:
                    logger.error(f"failed to save img from URL: {URL}")
        input_dir = __dir.joinpath("input", unique_folder_name)
        if run_predictions(input_dir, model, test_transform, device):
            target_dir = __dir.joinpath("input", "奶龙们", unique_folder_name)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            input_dir.rename(target_dir)
            await bot.call_api('delete_msg', message_id=event.message_id)
            await bot.set_group_ban(group_id = event.group_id, user_id = event.user_id, duration = 60)
            await seer.finish(MessageSegment.image(Path(__file__).parent / (str(randint(1,5)) + ".jpg")))
            # await seer.finish("发现奶龙")
        else:
            target_dir = __dir.joinpath("input", "非奶龙", unique_folder_name)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            input_dir.rename(target_dir)
            logger.info("没有发现奶龙")