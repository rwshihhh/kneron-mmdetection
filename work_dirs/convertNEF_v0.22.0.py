"""
v0.22.0 changes:
    1. The order of inference'result
    --> ktc.kneron_inference(input_data, onnx_file=onnx_path, input_names=["input"])
    
    2. The shape of element in img_list
    --> bie_model_path = km.analysis(input_mapping={"input": img_list})
"""

import ktc
import numpy as np
import os
import onnx
from PIL import Image
from os import walk
import time
# import version3_KL720DemoGenericInferenceYoloX_BypassHwPreProc

# Step 1.: set
platform = "630"
inf_platform = 630
model_id = 211
model_id_version = "0001"
#  8:  8-bit (default)
# 16: 16-bit
quantize_bits = 8 # set 8 or 16
onnx_path = "/put/your/onnxPATH.onnx"
optimized_onnx_path = "/put/your/opt.onnxPATH.opt.onnx"
quan_img_dir = "/put/yuor/qnan_img_dir"


# Step 2: Optimize the onnx model
m = onnx.load(onnx_path)
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
onnx.save(m, optimized_onnx_path)


# Step 3: Configure and load data necessary for ktc,
# and check if onnx is ok for toolchain
# npu (only) performance simulation
# Create a ModelConfig object. For details about this class, please check Appendix Python API.
# Here we set the model ID to 20008, the version to 0001 and the target platform to 720
km = ktc.ModelConfig(model_id, model_id_version, platform, onnx_model=m)
eval_result = km.evaluate()
print("\nNpu performance evaluation result:\n" + str(eval_result))


# Step 4: Quantize the onnx model
# prepare quan-img
img_list = []
for (dirpath, dirnames, filenames) in walk(quan_img_dir):
    for f in filenames:
        fullpath = os.path.join(dirpath, f)

        image = Image.open(fullpath)
        image = image.convert("RGB")
        image = Image.fromarray(np.array(image)[...,::-1])
        img_data = np.array(image.resize((640, 640), Image.BILINEAR)) / 256 - 0.5
        print(fullpath)
        img_data = np.transpose(img_data, (2, 0, 1))
        reshaped_array = img_data[np.newaxis, :]
        img_list.append(reshaped_array)
# fixed-point analysis
if quantize_bits == 8:
    """default (8-bit)"""
    bie_model_path = km.analysis(input_mapping={"input": img_list})
    # bie_model_path = km.analysis(input_mapping={"input": img_list}, quantize_mode = "post_sigmoid")
elif quantize_bits == 16:
    """16-bit"""
    #for v22:
    # don't set model_in/out as "16bit", it's only for debug and not work on chip.
    bie_model_path = km.analysis(
        input_mapping={"input": img_list},
        datapath_bitwidth_mode = "int16",
        weight_bitwidth_mode = "int16",
        model_in_bitwidth_mode = "int8",
        model_out_bitwidth_mode= "int8")
print("\nFixed-point analysis done. Saved bie model to '" + str(bie_model_path) + "'")


# Step 5: Compile
# The final step is to compile the BIE model into an NEF model.
# `compile` function takes a list of ModelConfig object.
# The `km` here is first defined in section 3 and quantized in section 4.
# The compiled binary file is saved in nef format. The path is returned as a `str` object.
nef_model_path = ktc.compile([km])
print("\nCompile done. Saved Nef file to '" + str(nef_model_path) + "'")
# import pdb; pdb.set_trace()
