DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."

# For distillation, the "teacher" is the Chapter 6 SFT model
# and the "student" is a LoRA adapter on the base model.
DEFAULT_TEACHER_DIR = "chapter06/runs/sft_run1"
