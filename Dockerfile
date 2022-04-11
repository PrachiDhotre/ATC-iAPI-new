FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-6
WORKDIR /
COPY trainer /trainer
RUN pip install -r /trainer/requirements.txt
ENTRYPOINT ["python", "-m", "trainer.ATC_iAPI_flasgger"]