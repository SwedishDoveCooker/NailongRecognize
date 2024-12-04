import TelegramBot from "node-telegram-bot-api";
import { HttpsProxyAgent } from "https-proxy-agent";
import fetch from "node-fetch";
import dotenv from "dotenv";
import FormData from "form-data";

dotenv.config();

const proxy = `http://${process.env.PROXY_HTTP_HOST}:${process.env.PROXY_HTTP_PORT}`;
const agent = new HttpsProxyAgent(proxy);

const bot = new TelegramBot(process.env.TELEGRAM_API_TOKEN, {
  polling: true,
  request: { agent: agent },
});

const API_URL = "http://10.10.12.184:7001/predict";

bot.on("message", async (msg) => {
  const chatId = msg.chat.id;

  if (msg.photo) {
    try {
      const fileId = msg.photo[msg.photo.length - 1].file_id;
      const file = await bot.getFile(fileId);
      const fileUrl = `https://api.telegram.org/file/bot${process.env.TELEGRAM_API_TOKEN}/${file.file_path}`;

     
      const response = await fetch(fileUrl, { agent : agent });
      const arrayBuffer = await response.arrayBuffer();

     
      const buffer = Buffer.from(arrayBuffer);

     
      const formData = new FormData();
      formData.append("image", buffer, {
        filename: "image.jpg",
        contentType: "image/jpeg",
      });

      const apiResponse = await fetch(API_URL, {
        method: "POST",
        body: formData,
        headers: formData.getHeaders(),
      });

      const apiResult = await apiResponse.json();

      if (apiResult.result) {
        bot.sendMessage(chatId, `预测结果: ${apiResult.result}`);
      } else {
        bot.sendMessage(chatId, "预测失败，请稍后重试。");
      }
    } catch (error) {
      console.error("处理图片时出错:", error);
      bot.sendMessage(chatId, "无法处理图片，请稍后重试。");
    }
  } else {
    bot.sendMessage(chatId, "请发送一张图片以进行预测。");
  }
});
