import sys
import os
import copy
import torch


argList = sys.argv[1:]

print(argList)

i = 0
commands = []
scales = []
types = []
models = []
input = "/content/drive/MyDrive/inputs/input.png"
output = None
compressOutput = True
inputFolder = None
outputFolder = None

#paths of the trained models for varius super resolution DL networks

model_dict = {
    "realesrgan": {
        "sr_path": "ESRGAN",
        "model_path": "experiments/pretrained_models/net_g_15000_realesrgan.pth"
    },
    "msrresnet": {
        "sr_path": "MSR_SWINIR",
        "model_path": "pretrained_weights/msrresnet_x4_psnr.pth"
    },
    "swinir": {
        "sr_path": "MSR_SWINIR",
        "model_path": "pretrained_weights/20000_G_swinIr.pth"
    },
    "HAT": {
        "sr_path": "HAT/hat",
        "model_path": "../experiments/pretrained_models/net_g_5000_hat.pth"
    }
}

#parsing the command line arguments
while i < len(argList):
    if argList[i] in {"-i","--input"}:
        input = argList[i+1]
        i+=2
    elif argList[i] in {"-o","--output"}:
        output = argList[i+1]
        i+=2
    elif argList[i] in {"-co","--compress_output"}:
        co = argList[i+1]
        if co=="False":
            compressOutput = False
        i+=2
    elif argList[i] in {"-if","--input_folder"}:
        inputFolder = argList[i+1]
        i+=2
    elif argList[i] in {"-of","--output_folder"}:
        outputFolder = argList[i+1]
        i+=2
    elif argList[i] in {"sr"}:
        srPath = None
        modelCode = "main_test_{}.py"
        model = "msrresnet"
        tile = None
        model_path = None
        folder_lq = ""
        file_name = ""
        output_path = ""
        scale = 4
        i+=1
        while i<len(argList) and argList[i].startswith("-"):
            if argList[i] == "--sr_path" or argList[i] == "-sp":
                srPath = argList[i+1]
            elif argList[i] == "--sr_model" or argList[i] == "-sm":
                model = argList[i+1]
            elif argList[i] == "--tile" or argList[i] == "-t":
                tile = int(argList[i+1])
            elif argList[i] == "--model_path" or argList[i] == "-mp":
                model_path = argList[i+1]
            elif argList[i] == "--scale" or argList[i] == "-s":
                scale = int(argList[i+1])
            else:
                i-=1
            i+=2
        if srPath is None:
            srPath = model_dict[model]["sr_path"]
        if model_path is None:
            model_path = model_dict[model]["model_path"]
        modelCode = modelCode.format(model)
        command = f"cd {srPath} && python {modelCode}"
        if tile is not None:
            command += f" --tile {tile}"
        if model_path is not None:
            command += f" --model_path {model_path}"
        command += f" --scale {scale}"
        command += f" --model_name {model}"
        types.append("sr")
        models.append(model)
        scales.append(scale)
        commands.append(command)
    elif argList[i] in {"int"}:
        inPath = "Interpolate/interpolate.py"
        scale = 4
        model = "bicubic"
        i+=1
        while i<len(argList) and argList[i].startswith("-"):
            if argList[i] == "--scale" or argList[i] == "-s":
                scale = int(argList[i+1])
            elif argList[i] == "--int_path" or argList[i] == "-ip":
                inPath = argList[i+1]
            elif argList[i] == "--int_model" or argList[i] == "-im":
                model = argList[i+1]
            else:
                i-=1
            i+=2
        command = f"python {inPath} --sf {scale}"
        types.append("int")
        models.append(model)
        scales.append(scale)
        commands.append(command)
    elif i<len(argList) and argList[i] in {"enh"}:
        enPath = "LightEnhancement"
        scale = 1
        model = "URetinex"
        i+=1
        while i<len(argList) and argList[i].startswith("-"):
            if argList[i] == "--scale" or argList[i] == "-s":
                _ = int(argList[i+1])
            elif argList[i] == "--enh_path" or argList[i] == "-ep":
                enPath = argList[i+1]
            elif argList[i] == "--enh_model" or argList[i] == "-em":
                model = argList[i+1]
            else:
                i-=1
            i+=2
        command = f"cd {enPath} && python test.py"
        types.append("enh")
        models.append(model)
        scales.append(scale)
        commands.append(command)
    elif i<len(argList) and argList[i] in {"shp"}:
        shpPath = "Sharpen_Denoise"
        scale = 1
        model = "blur"
        i+=1
        while i<len(argList) and argList[i].startswith("-"):
            if argList[i] == "--scale" or argList[i] == "-s":
                _ = int(argList[i+1])
            else:
                i-=1
            i+=2
        command = f"cd {shpPath} && python sharpen.py"
        types.append("shp")
        models.append(model)
        scales.append(scale)
        commands.append(command)
    elif i<len(argList) and argList[i] in ("den"):
        denPath = "Sharpen_Denoise/NAFNet"
        scale = 1
        model = "NAFNet"
        tile = None
        i+=1
        while i<len(argList) and argList[i].startswith("-"):
            if argList[i] == "--scale" or argList[i] == "-s":
                _ = int(argList[i+1])
            elif argList[i] == "--tile" or argList[i] == "-t":
                tile = int(argList[i+1])
            else:
                i-=1
            i+=2
        command = f"cd {denPath} && python denoise.py"
        if tile is not None:
            command += f" --tile {tile}"
        types.append("den")
        models.append(model)
        scales.append(scale)
        commands.append(command)

#processing a single file
if inputFolder is None:
    ext = input.split(".")[1]

    if outputFolder is None:
        outputFolder = "/content/drive/MyDrive/outputs/"

    outputFormat = outputFolder + "{}_{}_{}." + ext
    outputPaths = [outputFormat for _ in models]
    inputPaths = [outputFormat for _ in models]


    inputPaths[0] = input
    img_name, ext = os.path.splitext(os.path.basename(input))
    outputPaths[0] = outputFormat.format(img_name, models[0], scales[0])
    for i in range(1, len(outputPaths)):
        inputPaths[i] = outputPaths[i-1]
        img_name, ext = os.path.splitext(os.path.basename(inputPaths[i]))
        outputPaths[i] = outputFormat.format(img_name, models[i], scales[i])
    if output is not None:
        outputPaths[-1] = output

    rm_command = "rm -rf {}"

    for i in range(len(commands)):
        if types[i]=="enh":
            commands[i] += f" --img_path {inputPaths[i]}"
            commands[i] += f" --output {os.path.dirname(outputPaths[i])}"
            commands[i] += f" --output_path {outputPaths[i]}"
        elif types[i]=="sr":
            commands[i] += f" --folder_lq {os.path.dirname(inputPaths[i])}"
            commands[i] += f" --file_name {os.path.basename(inputPaths[i])}"
            commands[i] += f" --output_path {outputPaths[i]}"
        elif types[i]=="int":
            commands[i] += f" --in_path {inputPaths[i]}"
            commands[i] += f" --out_path {outputPaths[i]}"
        elif types[i]=="shp":
            commands[i] += f" --in_path {inputPaths[i]}"
            commands[i] += f" --out_path {outputPaths[i]}"
        elif types[i]=="den":
            commands[i] += f" --in_path {inputPaths[i]}"
            commands[i] += f" --out_path {outputPaths[i]}"
        
        if compressOutput and i>0:
            commands[i] += " && " + rm_command.format(inputPaths[i])

    for command in commands:
        print("Executing command")
        print(command)
        os.system(command)
    torch.cuda.empty_cache()
#processing all files in a folder
#PLEASE MAKE SURE ALL ARE IMAGES ONLY IF FOLDER IS PASSED
else:
    copy_commands = copy.deepcopy(commands)
    for file in os.listdir(inputFolder):
        commands = copy.deepcopy(copy_commands)

        input = os.path.join(inputFolder, file)

        ext = input.split(".")[1]

        if outputFolder is None:
            outputFolder = "/content/drive/MyDrive/outputs/"

        outputFormat = outputFolder + "{}_{}_{}." + ext
        outputPaths = [outputFormat for _ in models]
        inputPaths = [outputFormat for _ in models]


        inputPaths[0] = input
        img_name, ext = os.path.splitext(os.path.basename(input))
        outputPaths[0] = outputFormat.format(img_name, models[0], scales[0])
        for i in range(1, len(outputPaths)):
            inputPaths[i] = outputPaths[i-1]
            img_name, ext = os.path.splitext(os.path.basename(inputPaths[i]))
            outputPaths[i] = outputFormat.format(img_name, models[i], scales[i])
        if output is not None:
            outputPaths[-1] = output

        rm_command = "rm -rf {}"

        for i in range(len(commands)):
            if types[i]=="enh":
                commands[i] += f" --img_path {inputPaths[i]}"
                commands[i] += f" --output {os.path.dirname(outputPaths[i])}"
                commands[i] += f" --output_path {outputPaths[i]}"
            elif types[i]=="sr":
                commands[i] += f" --folder_lq {os.path.dirname(inputPaths[i])}"
                commands[i] += f" --file_name {os.path.basename(inputPaths[i])}"
                commands[i] += f" --output_path {outputPaths[i]}"
            elif types[i]=="int":
                commands[i] += f" --in_path {inputPaths[i]}"
                commands[i] += f" --out_path {outputPaths[i]}"
            elif types[i]=="shp":
                commands[i] += f" --in_path {inputPaths[i]}"
                commands[i] += f" --out_path {outputPaths[i]}"
            elif types[i]=="den":
                commands[i] += f" --in_path {inputPaths[i]}"
                commands[i] += f" --out_path {outputPaths[i]}"
            
            if compressOutput and i>0:
                commands[i] += " && " + rm_command.format(inputPaths[i])

        for command in commands:
            print("Executing command")
            print(command)
            os.system(command)
        torch.cuda.empty_cache()
