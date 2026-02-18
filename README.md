# ⚖️ The Historical Court: Multi-Agent System

ระบบ **Multi-Agent** จำลองการพิจารณาคดีทางประวัติศาสตร์ พัฒนาด้วย **Google ADK (Agent Development Kit)** โดยใช้โมเดล Gemini เพื่อวิเคราะห์ข้อมูลจาก Wikipedia ในรูปแบบ "ศาลจำลอง" (Court Simulation) เพื่อลดอคติ (Bias) และสรุปข้อเท็จจริงที่เป็นกลางที่สุด

---

## 🧩 Project Concept
โปรเจกต์นี้แก้ปัญหา **Information Bias** โดยการสร้าง AI Agents ที่มี "Persona" ขัดแย้งกัน 2 ฝั่ง (Admirer vs Critic) ให้ไปหาข้อมูลมาโต้แย้งกัน ก่อนจะมี Agent กลาง (Judge) คอยตัดสินว่าข้อมูลครบถ้วนพอหรือยัง แล้วจึงสรุปผล

### 🛠 Tech Stack
* **Framework:** Google ADK (Agent Development Kit)
* **Model:** Google Gemini Pro (via Vertex AI or API Key)
* **Tools:** LangChain (Wikipedia Wrapper), Python Standard Lib
* **Pattern:** Sequential, Parallel, and Loop Architectures

---

## 🏗️ System Architecture (โครงสร้างระบบ)

การทำงานแบ่งเป็น 4 Phase หลัก ตาม Flow นี้:

```mermaid
graph TD
    UserInput -->|Step 1| Clerk(The Inquiry)
    Clerk --> LoopStart
    subgraph "The Investigation Loop"
        LoopStart -->|Step 2| Team{Parallel Team}
        Team -->|Positive| Admirer(Agent A)
        Team -->|Negative| Critic(Agent B)
        Admirer -->|pos_data| Judge(Agent C)
        Critic -->|neg_data| Judge
        Judge -->|Decision| Check{Enough Info?}
        Check -- No -->|Feedback| Team
    end
    Check -- Yes -->|Step 4| Writer(Verdict Writer)
    Writer --> OutputFile[.txt Report]
