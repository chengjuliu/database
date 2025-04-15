import os, re, time, io, logging, warnings
import pandas as pd
from PIL import Image
import torch
import torchaudio
import pdfplumber
import spacy
from openai import OpenAI
from transformers import (
    CLIPProcessor, CLIPModel,
    WhisperProcessor, WhisperForConditionalGeneration
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# 📦 日志和警告抑制
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*CropBox.*')
warnings.filterwarnings('ignore', message='.*MediaBox.*')

# ✅ spaCy 中文模型
nlp = spacy.load("zh_core_web_sm")

# ✅ DeepSeek 配置
key = 'sk-6b461ad45209484b937b94b0693e6aa1'
api_url = "https://api.deepseek.com/v1"


def printChar(text, delay=0.05):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def sendToDeepSeek(say):
    client = OpenAI(api_key=key, base_url=api_url)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业的客服助手，请用正式的语气回答用户的问题。"},
            {"role": "user", "content": say},
        ],
        stream=False
    )
    return response.choices[0].message.content


# ✅ 向量模型
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ✅ PDF 按句拆分
def load_pdf_chunks(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    all_sentences = []
    sentence_id = 0
    for idx, doc in enumerate(docs):
        content = doc.page_content.replace("\n", "")
        content = re.sub(r'([。！？.!?])', r"\1##SPLIT##", content)
        sentences = [s.strip() for s in content.split("##SPLIT##") if s.strip()]
        page_num = doc.metadata.get("page", idx + 1)
        for i, sentence in enumerate(sentences):
            sentence_id += 1
            meta = {
                "source": f"page_{page_num}",
                "page": page_num,
                "sentence_number": i + 1,
                "global_id": sentence_id
            }
            all_sentences.append(Document(page_content=sentence, metadata=meta))
    return all_sentences


# ✅ 图像嵌入
def embed_images(image_dir):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    embeddings, image_count = [], 0
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_count += 1
            image = Image.open(os.path.join(image_dir, filename)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.get_image_features(**inputs)
            # 使用文本嵌入模型重新处理图像描述
            image_text = f"图像_{image_count}"
            image_embedding = embedder.embed_query(image_text)
            embeddings.append((image_text, image_embedding))
    return embeddings, image_count


# ✅ 音频转写
def embed_audio(audio_dir):
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    result = []
    for file in os.listdir(audio_dir):
        if file.endswith(".mp3") or file.endswith(".wav"):
            waveform, sr = torchaudio.load(os.path.join(audio_dir, file))
            inputs = processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt").input_features
            ids = model.generate(inputs)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            result.append((f"音频转录：{text}", text))
    return result


# ✅ 表格提取
def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables()
            for table in page_tables:
                if table and len(table) > 1:
                    header = [cell or "" for cell in table[0]]
                    body = [[cell or "" for cell in row] for row in table[1:] if any(row)]
                    if header and body:
                        total = len(header) * len(body)
                        filled = sum(1 for row in body for c in row if c.strip())
                        tables.append({
                            "page": page_num,
                            "table_num": len(tables) + 1,
                            "header": header,
                            "content": body,
                            "quality_score": {
                                "completeness": round(filled / total, 2) if total else 0,
                                "cell_count": total
                            }
                        })
    return tables


# ✅ 增强关系抽取
def extract_information(docs):
    extracted = {"文本": [], "公式": [], "关系": [], "图像": [], "表格": []}
    for doc in docs:
        text = doc.page_content
        if not isinstance(text, str):
            continue
        extracted["文本"].append(text)

        # 公式提取
        formulas = re.findall(r'[\w\d\(\)\+\-\*/=\^√∑π\[\]\{\}．]+', text)
        formulas = [f for f in formulas if any(op in f for op in ['=', '^', '√', '∑', 'π']) and len(f) > 5]
        extracted["公式"].extend(formulas)

        # 使用 spaCy 进行关系抽取
        parsed = nlp(text)
        for sent in parsed.sents:
            subj = [t for t in sent if 'subj' in t.dep_]
            obj = [t for t in sent if 'obj' in t.dep_ or t.dep_ == 'attr']
            root = [t for t in sent if t.dep_ == 'ROOT']
            ents = list(sent.ents)
            if subj and root and obj:
                extracted["关系"].append((subj[0].text, root[0].text, obj[0].text))
            elif len(ents) >= 2 and root:
                extracted["关系"].append((ents[0].text, root[0].text, ents[1].text))
    return extracted


# ✅ 构建向量库
def build_multimodal_vectorstore(pdf_path, image_dir, audio_dir, db_path):
    print("📘 加载 PDF...")
    docs = load_pdf_chunks(pdf_path)
    print("📊 提取表格...")
    tables = extract_tables_from_pdf(pdf_path)
    print("🖼️ 图像嵌入...")
    image_embeds, img_count = embed_images(image_dir)
    print("🔊 音频转写...")
    audio_embeds = embed_audio(audio_dir)

    # 将音频转写结果加入文档
    for text, _ in audio_embeds:
        docs.append(Document(page_content=text, metadata={"source": "audio"}))
    # 将表格信息加入文档
    for table in tables:
        rows = "\n".join(str(dict(zip(table['header'], r))) for r in table['content'][:3])
        docs.append(Document(
            page_content=f"表格-第{table['page']}页:\n表头:{table['header']}\n{rows}",
            metadata={"source": f"table_page_{table['page']}", "type": "table"}
        ))

    clean_docs = [Document(page_content=d.page_content.strip(), metadata=d.metadata)
                  for d in docs if d.page_content.strip()]
    print(f"✅ 文本条数：{len(clean_docs)}")

    # 用 PDF 文本构建基础向量库
    vs = FAISS.from_documents(clean_docs, embedder, normalize_L2=True)

    # 处理图像嵌入：一次性添加所有图像的文本和向量
    if image_embeds:
        # 将文本和向量组合成元组列表
        text_embeddings = [(text, vec) for text, vec in image_embeds]
        # 如果只有一条，为避免内部 zip 解包时将字符串拆分成字符，复制一份
        if len(text_embeddings) == 1:
            text_embeddings = text_embeddings * 2
        vs.add_embeddings(text_embeddings)
    vs.save_local(db_path)

    info = extract_information(clean_docs)
    info["图像"] = [f"图像_{i + 1}" for i in range(img_count)]
    info["表格"] = [f"{t['page']}页_表{t['table_num']}" for t in tables]

    print("\n📊 信息抽取统计：")
    df = pd.DataFrame({
        "类型": list(info.keys()),
        "数量": [len(v) for v in info.values()],
        "样例": [str(v[:2]) for v in info.values()]
    })
    print(df.to_string(index=False))


# ✅ 查询向量库
def query_vectorstore(query, db_path):
    vs = FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)
    results = vs.similarity_search(query, k=3)
    return "\n".join([f"{d.metadata.get('source', '')} 第{d.metadata.get('sentence_number', '?')}句：{d.page_content}"
                      for d in results])


# ✅ 主程序
if __name__ == "__main__":
    pdf_path = r"C:\Users\lcj\Desktop\test.pdf"
    image_dir = "./images"
    audio_dir = "./audios"
    db_path = "./mm_db"

    if not os.path.exists(db_path):
        build_multimodal_vectorstore(pdf_path, image_dir, audio_dir, db_path)

    while True:
        myin = input("您请说：")
        if myin.lower() == 'bye':
            print("欢迎下次使用！再见！")
            break
        related = query_vectorstore(myin, db_path)
        prompt = f"请根据以下资料回答问题：\n{related}\n\n问题：{myin}"
        answer = sendToDeepSeek(prompt)
        printChar(answer)
        print("-----------------------------------------------------------")