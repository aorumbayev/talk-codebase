import os
import requests
from tqdm import tqdm

import fire
import yaml

from talk_codebase.LLM import factory_llm
from talk_codebase.consts import DEFAULT_CONFIG

config_path = os.path.join(os.path.expanduser("~"), ".talk_codebase_config.yaml")

DEFAULT_MODEL_URL = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"


def get_config():
    print(f"🤖 Loading config from {config_path}:")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return config


def save_config(config):
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def download_model(url, path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)  # ensure the directory exists

    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            progress_bar.update(len(chunk))
            f.write(chunk)

    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("🤖 ERROR: Something went wrong while downloading the model")
    else:
        print(f"🤖 Model downloaded to {path}")


def configure():
    config = get_config()
    model_url = input(
        "🤖 Enter your custom model download URL or press Enter to use default: "
    )
    model_url = model_url if model_url else DEFAULT_MODEL_URL
    config["model_url"] = model_url
    download_model(model_url, DEFAULT_CONFIG["model_path"])
    save_config(config)
    print("🤖 Configuration saved!")


def loop(llm):
    while True:
        query = input("👉 ").lower().strip()
        if not query:
            print("🤖 Please enter a query")
            continue
        if query in ("exit", "quit"):
            break
        llm.send_query(query)


def validate_config(config):
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
    model_path = config.get("model_path")
    if not model_path:
        print("🤖 Please configure your model path. Use talk-codebase configure")
        exit(0)
    save_config(config)
    return config


def chat(root_dir):
    config = validate_config(get_config())
    llm = factory_llm(root_dir, config)
    loop(llm)


def main():
    try:
        fire.Fire({"chat": chat, "configure": configure})
    except KeyboardInterrupt:
        print("\n🤖 Bye!")
    except Exception as e:
        if str(e) == "<empty message>":
            print("🤖 Please configure your model path. Use talk-codebase configure")
        else:
            raise e


if __name__ == "__main__":
    main()
