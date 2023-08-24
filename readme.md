# tb project
## dependency
### must
1. docker
2. docker-compose
### optional
1. Vscode
### important packages
3. python3.10 (install in docker)
4. python's packages in requirements.txt (build in docker) 

## other files
1. data files (.pickle, not show in github)
2. paper draft and ppts (inside results folder)
3. etc (some envs)
4. docker files

## configuration
0. put the .pickle data in the folder
1. to build docker image and start jupyter server

    `docker-compose -f docker-compose.yaml up`

2. once built, visit jupyter lab through the url show console like

    `http://127.0.0.1:5000/lab?token=6b372517a0bbc447ed4702faa1b9a81675d4fad827e51ed2`

3. or u can directlly view the result in Vscode

## debug via Vscode
1. setting python interpreter
    - Open the Command Palette in Visual Studio Code (Ctrl+Shift+P or Cmd+Shift+P).
    - Run the "Remote-Containers: Attach to Running Container..." command.
    - Select the running Docker container with the Python interpreter you want to use (han_tb-app_1).
    - After attaching to the container,Vscode will refresh, then install python and pylance in extension in sidebar
    - Open the Command Palette in Visual Studio Code (Ctrl+Shift+P or Cmd+Shift+P).
    - Run the "Python: Select Interpreter"
    - enter /p_3_10/bin/python3