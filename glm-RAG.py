import os
import time
import json
import threading
import numpy as np
import torch
import faiss
from flask import Flask, request, jsonify, render_template, render_template_string
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

app = Flask(__name__)


# ------------------------
# 嵌入模型封装：使用本地 bertancientchinese 模型
# ------------------------
class BgeEncoder:
    def __init__(self, model_path: str, batch_size: int = 512):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        print(f"初始化嵌入模型到设备: {self.device.upper()}")

    def encode(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 使用 [CLS] 向量作为文本表示
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy().astype("float32")


# ------------------------
# 本地大模型封装（DeepSeek-1.3B）
# ------------------------
class LocalLLM:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载本地大模型，路径: {model_path}，设备: {self.device.upper()}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_length: int = 1024, temperature: float = 0.3) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


# ------------------------
# 配置参数
# ------------------------
class Config:
    def __init__(self):
        # 仅使用 Q&A.txt 文件作为知识库
        self.data_files = [r"Q&A.txt"]
        self.top_k = 5
        self.similarity_threshold = 0.4
        self.batch_size = 512
        # 文字嵌入模型路径（请将该路径指向您的 bertancientchinese 模型所在目录）
        self.embedding_model_path = r"C:\Users\cy\RAG\model"
        # 此处 prompt_instruction 在本项目不再使用（前端回答中不展示），可保留说明性文字
        self.prompt_instruction = "【仅依据数据库记录回答】"


config = Config()


# ------------------------
# 向量搜索引擎
# ------------------------
class VectorSearchEngine:
    def __init__(self, encoder: BgeEncoder, similarity_threshold: float):
        self.encoder = encoder
        self.index = None
        self.data = []
        self.similarity_threshold = similarity_threshold

    def build_index(self, items):
        if not items:
            print("⚠️ 无数据可用于构建索引")
            return
        self.index = None
        self.data = []
        total = len(items)
        print(f"🛠 开始构建索引，共 {total} 条数据...")
        for i in range(0, total, self.encoder.batch_size):
            batch = items[i:i + self.encoder.batch_size]
            self._add_batch(batch)
            print(f"⏳ 已处理 {min(i + self.encoder.batch_size, total)}/{total} 条")
        print(f"✅ 索引构建完成，总数据量：{len(self.data)} 条")

    def _add_batch(self, batch):
        texts = [item["content"] for item in batch]
        embeddings = self.encoder.encode(texts)
        faiss.normalize_L2(embeddings)
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.data.extend(batch)

    def search(self, query: str, top_k: int = 5):
        start_time = time.time()
        if self.index is None:
            print("⚠️ 请先构建索引")
            return []
        query_vec = self.encoder.encode([query])
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, top_k * 2)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or score < self.similarity_threshold:
                continue
            item = self.data[idx].copy()
            item["similarity"] = float(score)
            item["record_index"] = int(idx)
            results.append(item)
        if len(results) > top_k:
            results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
            dynamic_threshold = results[-1]["similarity"] * 0.9
            results = [r for r in results if r["similarity"] >= dynamic_threshold]
        print(f"🔍 搜索耗时：{time.time() - start_time:.2f}s，找到 {len(results)} 条结果")
        return results


# ------------------------
# 解析问答数据库文件：每条记录格式为 "问题：xxx\n答案：yyy"
# ------------------------
def parse_qa_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    items = []
    question, answer = None, None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("问题："):
            if question and answer:
                content = f"{question}\n{answer}"
                items.append({"content": content, "metadata": {}})
            question = line
            answer = ""
        elif line.startswith("答案："):
            answer = line
        else:
            if answer != "":
                answer += "\n" + line
            elif question is not None:
                question += "\n" + line
    if question and answer:
        content = f"{question}\n{answer}"
        items.append({"content": content, "metadata": {}})
    print(f"已解析 {file_path}，共获得 {len(items)} 条问答记录")
    return items


# ------------------------
# 知识库：加载数据文件并构建知识库
# ------------------------
class QAKnowledgeBase:
    def __init__(self, data_files, embedding_model_path, similarity_threshold, batch_size):
        self.data_files = data_files
        self.encoder = BgeEncoder(embedding_model_path, batch_size=batch_size)
        self.search_engine = VectorSearchEngine(self.encoder, similarity_threshold)
        self.load_data()

    def load_data(self):
        all_items = []
        for file_path in self.data_files:
            items = parse_qa_file(file_path)
            if items:
                all_items.extend(items)
        if all_items:
            self.search_engine.build_index(all_items)
        else:
            print("⚠️ 无有效数据加载")

    def refresh(self):
        print("🔄 正在刷新知识库数据...")
        self.load_data()
        print("✅ 知识库数据已更新。")

    def query(self, question: str):
        return self.search_engine.search(question, config.top_k)


# ------------------------
# 辅助函数：判断数据库记录中的问题与用户问题是否匹配
# ------------------------
def is_similar_question(db_question: str, user_question: str) -> bool:
    def normalize(q):
        return q.replace("？", "?").replace(" ", "").strip()

    return normalize(db_question) in normalize(user_question) or normalize(user_question) in normalize(db_question)


# ------------------------
# 初始化知识库（仅使用 Q&A.txt 作为数据来源）
# ------------------------
qa_knowledge_base = QAKnowledgeBase(config.data_files, config.embedding_model_path, config.similarity_threshold,
                                    config.batch_size)


# 此处大模型接口未使用


# ------------------------
# 后台定时刷新知识库（每5分钟刷新一次）
# ------------------------
def periodic_refresh(interval=300):
    while True:
        time.sleep(interval)
        try:
            qa_knowledge_base.refresh()
        except Exception as e:
            print(f"刷新知识库出错: {e}")


refresh_thread = threading.Thread(target=periodic_refresh, daemon=True)
refresh_thread.start()


# ------------------------
# 前端页面（保持原有前端代码不变）
# ------------------------
@app.route("/")
def index():
    return render_template_string(r'''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>古籍智能问答系统</title>
        <link href="https://cdn.staticfile.org/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary-color: #2c3e50;
                --secondary-color: #3498db;
                --background-color: #f8f9fa;
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            body {
                font-family: "Microsoft YaHei", sans-serif;
                background: var(--background-color);
                display: flex;
                min-height: 100vh;
            }
            .sidebar {
                width: 280px;
                background: white;
                padding: 20px;
                box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            }
            .main-content {
                flex: 1;
                padding: 30px;
                max-width: 1200px;
            }
            .card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .status-bar {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 25px;
            }
            .status-item {
                padding: 15px;
                border-radius: 8px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                resize: vertical;
                font-size: 16px;
            }
            button {
                background: var(--secondary-color);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
            }
            button:hover {
                opacity: 0.9;
                transform: translateY(-1px);
            }
            .example-questions {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin: 20px 0;
            }
            .answer-container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-top: 25px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <!-- 侧边栏 -->
        <div class="sidebar">
            <div class="card">
                <h2><i class="fas fa-cog"></i> 系统配置</h2>
                <div style="margin-top:15px">
                    <label>选择大模型：</label>
                    <select id="modelSelect" style="width:100%;padding:8px;margin-top:5px">
                        <option value="DeepSeek-1.3B">DeepSeek-1.3B</option>
                        <option value="ChatGLM3-6B">ChatGLM2-6b</option>
                        <option value="Qwen-1.8B">Qwen-1.8B</option>
                    </select>
                </div>
            </div>

            <div class="card" style="margin-top:20px">
                <h3><i class="fas fa-database"></i> 知识库状态</h3>
                <div id="knowledgeStatus" style="margin-top:10px;color:#666">
                    加载中...
                </div>
            </div>

            <div class="card" style="margin-top:20px">
                <h3><i class="fas fa-chart-bar"></i> 性能监控</h3>
                <div id="gpuStatus" style="margin-top:10px;color:#666">
                    GPU内存: 加载中...
                </div>
            </div>
        </div>

        <!-- 主内容 -->
        <div class="main-content">
            <h1 style="color:var(--primary-color);margin-bottom:25px">
                <i class="fas fa-scroll"></i> 古籍文献智能问答系统
            </h1>

            <!-- 状态栏 -->
            <div class="status-bar">
                <div class="status-item">
                    <div>当前模型</div>
                    <div id="currentModel" style="font-weight:bold;color:var(--secondary-color)">-</div>
                </div>
                <div class="status-item">
                    <div>检索模式</div>
                    <div style="font-weight:bold;color:#27ae60">混合检索</div>
                </div>
                <div class="status-item">
                    <div>Embedding模型</div>
                    <div style="font-weight:bold;color:#e67e22">古文BERT</div>
                </div>
            </div>

            <!-- 问题输入 -->
            <textarea id="questionInput" rows="4" placeholder="请输入您的问题，例如：《道德经》中'道可道非常道'如何解释？"></textarea>

            <!-- 示例问题 -->
            <div class="example-questions">
                <button onclick="setExample('孔子认为学习的快乐和君子的修养体现在哪些方面？')">示例问题1</button>
                <button onclick="setExample('《诗经》的主要艺术特色有哪些？')">示例问题2</button>
                <button onclick="setExample('解释《孟子》中的民为贵思想')">示例问题3</button>
            </div>

            <button onclick="submitQuestion()" style="width:100%;padding:15px;font-size:16px">
                <i class="fas fa-paper-plane"></i> 提交问题
            </button>

            <!-- 回答区域 -->
            <div id="answerSection" style="display:none;margin-top:30px">
                <div class="card">
                    <h3><i class="fas fa-comment-dots"></i> 智能回答</h3>
                    <div id="answerContent" class="answer-container" style="margin-top:15px"></div>
                </div>

                <div class="card" style="margin-top:20px">
                    <h3><i class="fas fa-book-open"></i> 参考来源</h3>
                    <div id="sourceDocuments" style="margin-top:15px"></div>
                </div>
            </div>
        </div>

        <script>
            // 初始化状态
            function updateStatus() {
                // 更新模型选择
                const modelSelect = document.getElementById('modelSelect');
                document.getElementById('currentModel').textContent = modelSelect.value;

                // 获取系统状态（示例数据）
                fetch('/get_status').then(r => r.json()).then(data => {
                    document.getElementById('knowledgeStatus').innerHTML = `
                        已加载文档：${data.doc_count}条<br>
                        数据来源：${data.sources.join(', ')}
                    `;
                    document.getElementById('gpuStatus').textContent = 
                        `GPU内存：${data.gpu_mem}GB | 处理速度：${data.speed}ms`;
                });
            }

            // 示例问题填充
            function setExample(question) {
                document.getElementById('questionInput').value = question;
            }

            // 提交问题
            function submitQuestion() {
                const question = document.getElementById('questionInput').value.trim();
                if (!question) return alert('请输入问题内容');

                const answerSection = document.getElementById('answerSection');
                const answerContent = document.getElementById('answerContent');
                const sources = document.getElementById('sourceDocuments');

                answerSection.style.display = 'block';
                answerContent.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 正在生成回答...';
                sources.innerHTML = '';

                fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ question: question })
                }).then(r => r.json()).then(data => {
                    answerContent.innerHTML = data.answer;
                    if(data.sources) {
                        sources.innerHTML = data.sources.map((s, i) => `
                            <div class="source-card" style="margin:10px 0;padding:10px;border-left:3px solid #3498db">
                                <b>来源 ${i+1}</b><br>
                                ${s.content}<br>
                                <small style="color:#666">${s.metadata}</small>
                            </div>
                        `).join('');
                    }
                }).catch(e => {
                    answerContent.innerHTML = '请求失败，请重试';
                    console.error(e);
                });
            }

            // 初始状态更新
            updateStatus();
            setInterval(updateStatus, 5000); // 每5秒更新状态
        </script>
    </body>
    </html>
    ''')


# ------------------------
# 问答接口：直接从 Q&A.txt 知识库中找出该问题的所有答案，并返回
# ------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"answer": "问题为空，请重新输入。"})

    # 检索知识库中的记录（此处仅使用 Q&A.txt 中的记录）
    results = qa_knowledge_base.query(user_question)
    if not results:
        return jsonify({"answer": "未在知识库中找到相关记录，请核对问题描述。", "sources": []})

    # 筛选出问题部分与用户问题相近的所有记录
    matching = []
    for item in results:
        first_line = item["content"].split("\n")[0]
        if first_line.startswith("问题："):
            db_question = first_line[len("问题："):].strip()
            if is_similar_question(db_question, user_question):
                matching.append(item)

    if not matching:
        return jsonify({"answer": "未在知识库中找到与问题完全匹配的记录。", "sources": []})

    # 对于所有匹配记录，提取"答案："后面的内容，并用分隔符拼接
    all_answers = []
    for item in matching:
        content = item.get("content", "")
        idx = content.find("答案：")
        if idx != -1:
            answer_text = content[idx + len("答案："):].strip()
            all_answers.append(answer_text)
        else:
            all_answers.append(content.strip())

    final_answer = "\n\n".join(all_answers)

    # 同时返回所有匹配记录作为"参考来源"
    sources = []
    for item in matching:
        sources.append({
            "record_index": int(item.get("record_index", -1)),
            "content": item.get("content", ""),
            "metadata": ""  # 此处可以添加额外描述，如来源信息
        })

    # 返回的 JSON 只包含 answer 和 sources 两个字段，不展示 instruction、question、retrieved_context 等
    return jsonify({
        "answer": final_answer,
        "sources": sources
    })


@app.route("/get_status")
def get_status():
    return jsonify({
        "doc_count": len(qa_knowledge_base.search_engine.data),
        "sources": list(set([os.path.basename(f) for f in config.data_files])),
        "gpu_mem": f"{torch.cuda.memory_allocated() / 1e9:.1f}" if torch.cuda.is_available() else "N/A",
        "speed": "42.1"  # 示例值
    })


# ------------------------
# 启动 Flask 服务器
# ------------------------
if __name__ == "__main__":
    print("=== 环境检测 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"显存总量: {total_mem:.2f} GB")
    app.run(host="0.0.0.0", port=5000, debug=True)
