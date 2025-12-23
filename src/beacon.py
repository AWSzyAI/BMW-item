import os
from openai import OpenAI
import httpx
from dotenv import load_dotenv


# 从项目根目录加载 .env 中的环境变量
load_dotenv()


def ask_Qwen2(messages):
    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    aliyun_api_key = os.getenv("QWEN_ALIYUN_API_KEY")

    if not api_key or not base_url or not aliyun_api_key:
        raise RuntimeError("QWEN_API_KEY / QWEN_BASE_URL / QWEN_ALIYUN_API_KEY 未在环境变量中配置")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(verify=False, timeout=30),
        default_headers={
            "apikey": aliyun_api_key,
            "Content-Type": "application/json",
        },
    )

    try:
        # 调试：打印原始 messages
        print("[QWEN][REQUEST] messages =")
        try:
            import json as _json
            print(_json.dumps(messages, ensure_ascii=False, indent=2))
        except Exception:
            print(messages)

        response = client.chat.completions.create(
            model="Qwen2",
            messages=messages,
            temperature=0,
            max_tokens=8000,
            stream=False,
        )
        # 调试：打印原始 resp 对象（尽量可读）
        try:
            print("[QWEN][RAW_RESPONSE] =", response)
        except Exception:
            pass
        if hasattr(response, "choices") and len(response.choices) > 0:
            if hasattr(response.choices[0].message, "content"):
                content = response.choices[0].message.content
                print("[QWEN][CONTENT] =")
                print(content)
                return content
            else:
                print("响应中缺少content字段")
                return ""
        else:
            print("响应中缺少choices字段")
            return ""
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return ""


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "你是一个专业的SQL优化专家，请优化以下SQL语句："},
        {"role": "user", "content": "SELECT * FROM users WHERE id = 1"},
    ]
    ask_Qwen2(messages)