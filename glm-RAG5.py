import os
import time
import json
import threading
import numpy as np
import torch
import faiss
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModel
import requests

app = Flask(__name__)


# ------------------------
# 升级配置参数
# ------------------------
class EnhancedConfig:
    def __init__(self):
        self.data_files = [r"Q&A.txt"]
        self.top_k = 5
        self.similarity_threshold = 0.4
        self.batch_size = 512
        # 嵌入模型路径
        self.embedding_model_path = r"C:/Users/cy/.cache/modelscope/hub/models/dienstag/chinese-bert-wwm-ext"
        # DeepSeek API密钥
        self.deepseek_api_key = "sk-0ae5870ea20141149406634adc7fdba1"
        # RAG参数
        self.max_context_length = 1500
        self.llm_temperature = 0.3
        self.llm_max_length = 1024
        self.rag_prompt_template = """你是古籍问答助手，请基于以下参考内容回答用户的问题。

参考内容：
{context}

用户问题：{question}
请用中文详细回答。如果没有明确相关内容，请说"无相关资料"。"""


config = EnhancedConfig()


# ------------------------
# 模型加载（去除了accelerate依赖）
# ------------------------
class EnhancedModels:
    def __init__(self):
        # 嵌入模型
        self.embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_tokenizer = AutoTokenizer.from_pretrained(config.embedding_model_path)
        self.embed_model = AutoModel.from_pretrained(config.embedding_model_path).to(self.embed_device)
        print(f"✅ 嵌入模型加载成功 | 设备: {self.embed_device}")


models = EnhancedModels()


# ------------------------
# 增强版检索系统（保持不变）
# ------------------------
class EnhancedRetriever:
    def __init__(self):
        self.index = None
        self.data = []
        self._build_index()

    def _build_index(self):
        all_items = []
        for file_path in config.data_files:
            items = self.parse_qa_file(file_path)
            all_items.extend(items)

        if all_items:
            texts = [item["content"] for item in all_items]
            embeddings = self._batch_encode(texts)

            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            self.data = all_items
            print(f"🛠 索引构建完成 | 数据量: {len(all_items)}")
        else:
            raise ValueError("无有效数据构建索引")

    def parse_qa_file(self, file_path):
        items = []
        question, answer = None, None
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("问题："):
                    question = line.replace("问题：", "").strip()
                elif line.startswith("答案："):
                    answer = line.replace("答案：", "").strip()
                    if question and answer:
                        items.append({
                            "content": f"问题：{question}\n答案：{answer}",
                            "metadata": {"source": file_path},
                            "question": question,
                            "answer": answer
                        })
                        question, answer = None, None
        return items

    def _batch_encode(self, texts):
        inputs = models.embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(models.embed_device)

        with torch.no_grad():
            outputs = models.embed_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def search(self, query, top_k=5):
        query_vec = self._batch_encode([query])
        faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, top_k * 2)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx >= 0 and score >= 0.5:
                results.append({
                    "content": self.data[idx]["content"],
                    "question": self.data[idx]["question"],
                    "answer": self.data[idx]["answer"],
                    "score": float(score),
                    "index": int(idx)
                })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


retriever = EnhancedRetriever()


# ------------------------
# 生成系统（直接调用API生成）
# ------------------------
class EnhancedGenerator:
    def generate(self, question):
        # 直接调用DeepSeek API，不提供上下文
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.deepseek_api_key}"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": question}],
            "temperature": config.llm_temperature,
            "max_tokens": config.llm_max_length
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            return answer
        except requests.RequestException as e:
            print(f"API请求失败: {e}")
            return "系统调用API时发生错误"


generator = EnhancedGenerator()


# ------------------------
# 后端接口
# ------------------------
@app.route("/ask", methods=["POST"])
def enhanced_ask():
    start_time = time.time()
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "问题内容为空", "sources": []})

    try:
        # 获取搜索结果用于显示参考来源
        search_results = retriever.search(question, config.top_k)

        # 直接调用大模型生成答案
        answer = generator.generate(question)

        # 格式化参考来源
        sources = [{
            "record_index": res["index"],
            "content": f"问题：{res['question']}\n答案：{res['answer']}",
            "metadata": f"相似度: {res['score']:.2f} | 来源文件: {os.path.basename(config.data_files[0])}"
        } for res in search_results]

        return jsonify({
            "answer": answer,
            "sources": sources,
            "time_cost": round(time.time() - start_time, 2)
        })

    except Exception as e:
        print(f"处理失败: {str(e)}")
        return jsonify({"answer": "系统处理时发生错误", "sources": []})


@app.route("/get_status")
def get_status():
    return jsonify({
        "doc_count": len(retriever.data),
        "qa_count": len(retriever.data),
        "sources": list(set([os.path.basename(f) for f in config.data_files])),
        "gpu_mem": f"{torch.cuda.memory_allocated() / 1e9:.1f}" if torch.cuda.is_available() else "N/A",
        "speed": f"{torch.cuda.memory_allocated() / 1e6:.1f}MB"  # 示例值
    })


# ------------------------
# 整合后的前端界面
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
            /* 新增的模型切换动画样式 */
            .model-switch-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
                display: none;
            }
            .model-switch-content {
                background: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .model-switch-spinner {
                font-size: 40px;
                margin-bottom: 20px;
                color: var(--secondary-color);
            }
            .model-switch-text {
                font-size: 18px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <!-- 模型切换遮罩层 -->
        <div class="model-switch-overlay" id="modelSwitchOverlay">
            <div class="model-switch-content">
                <div class="model-switch-spinner">
                    <i class="fas fa-spinner fa-spin"></i>
                </div>
                <div class="model-switch-text">
                    模型切换中，请稍候...
                </div>
            </div>
        </div>

        <!-- 侧边栏 -->
        <div class="sidebar">
            <div class="card">
                <h2><i class="fas fa-cog"></i> 系统配置</h2>
                <div style="margin-top:15px">
                    <label>选择大模型：</label>
                    <select id="modelSelect" style="width:100%;padding:8px;margin-top:5px" onchange="handleModelChange()">
                        <option value="DeepSeek">DeepSeek-1.3B</option>
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
                    <div style="font-weight:bold;color:#e67e22">chinese-bert-wwm-ext</div>
                </div>
            </div>

            <!-- 问题输入 -->
            <textarea id="questionInput" rows="4" placeholder="请输入您的问题，例如：《道德经》中'道可道非常道'如何解释？"></textarea>

            <!-- 示例问题 -->
            <div class="example-questions">
                <button onclick="setExample('孔子认为学习的快乐和君子的修养体现在哪些方面？')">示例问题1</button>
                <button onclick="setExample('有子如何论述孝悌与“仁”的关系？')">示例问题2</button>
                <button onclick="setExample('王阳明"格物致知"说的核心要义是什么？')">示例问题3</button>
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
                        问答条数：${data.qa_count} 条<br>
                        数据来源：${data.sources.join(', ')}
                    `;
                    document.getElementById('gpuStatus').textContent = 
                        `GPU内存：${data.gpu_mem} | 处理速度：${data.speed}`;
                });
            }

            // 示例问题填充
            function setExample(question) {
                document.getElementById('questionInput').value = question;
            }

            // 处理模型切换
            function handleModelChange() {
                const overlay = document.getElementById('modelSwitchOverlay');
                overlay.style.display = 'flex';

                // 模拟模型切换过程（实际应用中这里可以发送请求到后端切换模型）
                setTimeout(() => {
                    overlay.style.display = 'none';
                    updateStatus(); // 更新状态显示新模型
                }, 2000); // 2秒后关闭遮罩
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
                    if(data.sources && data.sources.length > 0) {
                        sources.innerHTML = data.sources.map((s, i) => `
                            <div class="source-card" style="margin:10px 0;padding:10px;border-left:3px solid #3498db">
                                <b>来源 ${i+1}</b><br>
                                ${s.content}<br>
                                <small style="color:#666">${s.metadata}</small>
                            </div>
                        `).join('');
                    } else {
                        sources.innerHTML = '<div style="color:#666">未找到相关参考内容</div>';
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
# 定时刷新线程（保持知识库更新）
# ------------------------
def periodic_refresh(interval=300):
    while True:
        time.sleep(interval)
        try:
            retriever._build_index()
            print("✅ 知识库已刷新")
        except Exception as e:
            print(f"刷新失败: {e}")


refresh_thread = threading.Thread(target=periodic_refresh, daemon=True)
refresh_thread.start()

if __name__ == "__main__":
    # 验证测试
    sample_input = "测试模型是否加载成功"
    test_vector = retriever._batch_encode([sample_input])
    print(f"嵌入测试输出shape: {test_vector.shape}")

    test_output = generator.generate("测试生成")
    print(f"生成测试结果: {test_output}")

    app.run(host="0.0.0.0", port=5000, debug=True)