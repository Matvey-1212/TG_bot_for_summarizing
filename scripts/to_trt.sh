trtexec --onnx=/TG_bot_for_summarizing/scripts/onnx_model/mbart_model.onnx \
        --saveEngine=/TG_bot_for_summarizing/weights/engine/mbart_model.engine \
        --fp16 \
        --minShapes=input_name:1 \
        --optShapes=input_name:8 \
        --maxShapes=input_name:32
