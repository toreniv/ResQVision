# ResQVision Demo Steps

## Full Local Demo

1. Open VS Code.
2. Open a new terminal.
3. Verify CUDA environment:
   where cl
   nvcc --version

4. Run CUDA data pipeline:
   .\run_data_pipeline.ps1

5. Start YOLO Live:
   .\venv\Scripts\python.exe scripts\yolo_live.py

6. Start frontend:
   cd frontend
   npm run dev

7. Open:
   http://localhost:5173

## Present Pages

- Mission Plan
- Tactical Command
- Analytics
- Computer Vision
- System Architecture

## Fallbacks

If local CUDA fails:
- Use Colab ZIP import.

If YOLO Live fails:
- Use existing detections.json / detection_preview.jpg.
- The dashboard still works with risk_ranking.json.

If frontend fails:
- Run npm install inside frontend.