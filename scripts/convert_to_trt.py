{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[03/23/2025-00:19:19] [TRT] [W] ModelImporter.cpp:459: Make sure input input_ids has Int64 binding.\n",
      "[03/23/2025-00:19:19] [TRT] [W] ModelImporter.cpp:459: Make sure input attention_mask has Int64 binding.\n",
      "Движок сохранен в /home/matvey.antonov@factory.vocord.ru/main/temp/TG_bot_for_summarizing/weights/engine/encoder_model.trt\n",
      "[03/23/2025-00:19:26] [TRT] [W] ModelImporter.cpp:459: Make sure input encoder_attention_mask has Int64 binding.\n",
      "[03/23/2025-00:19:26] [TRT] [W] ModelImporter.cpp:459: Make sure input input_ids has Int64 binding.\n",
      "[03/23/2025-00:19:26] [TRT] [E] WeightsContext.cpp:178: Failed to open file: decoder_model.onnx_data\n",
      "In node -1 with name:  and operator:  (parseGraph): INVALID_GRAPH: Failed to import initialzer\n",
      "[03/23/2025-00:19:26] [TRT] [E] In node -1 with name:  and operator:  (parseGraph): INVALID_GRAPH: Failed to import initialzer\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Ошибка при парсинге ONNX-модели",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[2], line 62\u001b[0m\n\u001b[1;32m     60\u001b[0m decoder_onnx_path \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m/home/matvey.antonov@factory.vocord.ru/main/temp/TG_bot_for_summarizing/weights/onnx/decoder_model.onnx\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m     61\u001b[0m decoder_engine_path \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m/home/matvey.antonov@factory.vocord.ru/main/temp/TG_bot_for_summarizing/weights/engine/decoder_model.trt\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m---> 62\u001b[0m \u001b[43mbuild_engine\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdecoder_onnx_path\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdecoder_engine_path\u001b[49m\u001b[43m)\u001b[49m\n",
      "Cell \u001b[0;32mIn[2], line 21\u001b[0m, in \u001b[0;36mbuild_engine\u001b[0;34m(onnx_path, engine_path)\u001b[0m\n\u001b[1;32m     19\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m error \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mrange\u001b[39m(parser\u001b[38;5;241m.\u001b[39mnum_errors):\n\u001b[1;32m     20\u001b[0m         \u001b[38;5;28mprint\u001b[39m(parser\u001b[38;5;241m.\u001b[39mget_error(error))\n\u001b[0;32m---> 21\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mОшибка при парсинге ONNX-модели\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m     23\u001b[0m \u001b[38;5;66;03m# Настройка конфигурации движка\u001b[39;00m\n\u001b[1;32m     24\u001b[0m config \u001b[38;5;241m=\u001b[39m builder\u001b[38;5;241m.\u001b[39mcreate_builder_config()\n",
      "\u001b[0;31mValueError\u001b[0m: Ошибка при парсинге ONNX-модели"
     ]
    }
   ],
   "source": [
    "import tensorrt as trt\n",
    "\n",
    "# Инициализация TensorRT\n",
    "TRT_LOGGER = trt.Logger(trt.Logger.WARNING)\n",
    "EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)\n",
    "\n",
    "# Функция для создания движка TensorRT из ONNX-модели\n",
    "def build_engine(onnx_path, engine_path):\n",
    "    # Создание движка TensorRT\n",
    "    builder = trt.Builder(TRT_LOGGER)\n",
    "    network = builder.create_network(EXPLICIT_BATCH)\n",
    "    parser = trt.OnnxParser(network, TRT_LOGGER)\n",
    "\n",
    "    # Загрузка ONNX-модели\n",
    "    with open(onnx_path, \"rb\") as f:\n",
    "        onnx_model = f.read()\n",
    "\n",
    "    if not parser.parse(onnx_model):\n",
    "        for error in range(parser.num_errors):\n",
    "            print(parser.get_error(error))\n",
    "        raise ValueError(\"Ошибка при парсинге ONNX-модели\")\n",
    "\n",
    "    # Настройка конфигурации движка\n",
    "    config = builder.create_builder_config()\n",
    "    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)  # 6 ГБ\n",
    "\n",
    "    # Добавление оптимизационного профиля для динамических входов\n",
    "    profile = builder.create_optimization_profile()\n",
    "\n",
    "    # Установка диапазонов для динамических входов\n",
    "    for i in range(network.num_inputs):\n",
    "        input_tensor = network.get_input(i)\n",
    "        input_name = input_tensor.name\n",
    "        shape = input_tensor.shape\n",
    "        min_shape = (1, 1)    # Минимальные размеры (batch_size=1, seq_len=1)\n",
    "        opt_shape = (8, 128)  # Оптимальные размеры (batch_size=8, seq_len=128)\n",
    "        max_shape = (16, 512) # Максимальные размеры (batch_size=16, seq_len=512)\n",
    "        profile.set_shape(input_name, min_shape, opt_shape, max_shape)\n",
    "\n",
    "    config.add_optimization_profile(profile)\n",
    "\n",
    "    # Построение сериализованного движка\n",
    "    serialized_engine = builder.build_serialized_network(network, config)\n",
    "\n",
    "    if serialized_engine is None:\n",
    "        raise RuntimeError(\"Не удалось построить движок TensorRT\")\n",
    "\n",
    "    # Сохранение движка\n",
    "    with open(engine_path, \"wb\") as f:\n",
    "        f.write(serialized_engine)\n",
    "\n",
    "    print(f\"Движок сохранен в {engine_path}\")\n",
    "\n",
    "# Конвертация encoder\n",
    "encoder_onnx_path = \"/home/matvey.antonov@factory.vocord.ru/main/temp/TG_bot_for_summarizing/weights/onnx/encoder_model.onnx\"\n",
    "encoder_engine_path = \"/home/matvey.antonov@factory.vocord.ru/main/temp/TG_bot_for_summarizing/weights/engine/encoder_model.trt\"\n",
    "build_engine(encoder_onnx_path, encoder_engine_path)\n",
    "\n",
    "# Конвертация decoder\n",
    "decoder_onnx_path = \"/home/matvey.antonov@factory.vocord.ru/main/temp/TG_bot_for_summarizing/weights/onnx/decoder_model.onnx\"\n",
    "decoder_engine_path = \"/home/matvey.antonov@factory.vocord.ru/main/temp/TG_bot_for_summarizing/weights/engine/decoder_model.trt\"\n",
    "build_engine(decoder_onnx_path, decoder_engine_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"color: #800080; text-decoration-color: #800080; font-weight: bold\">Simplified model larger than 2GB. Trying to save as external data...</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1;35mSimplified model larger than 2GB. Trying to save as external data\u001b[0m\u001b[1;35m...\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mThe Kernel crashed while executing code in the the current cell or a previous cell. Please review the code in the cell(s) to identify a possible cause of the failure. Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details."
     ]
    }
   ],
   "source": [
    "import onnx\n",
    "from onnxsim import simplify\n",
    "\n",
    "# Загрузка модели\n",
    "model = onnx.load(decoder_onnx_path)\n",
    "\n",
    "# Упрощение модели\n",
    "simplified_model, check = simplify(model)\n",
    "assert check, \"Модель не удалось упростить\"\n",
    "\n",
    "# Сохранение упрощенной модели\n",
    "onnx.save(simplified_model, \"/home/matvey.antonov@factory.vocord.ru/main/temp/TG_bot_for_summarizing/weights/onnx/simplified_decoder_model.onnx\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dl_project",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
