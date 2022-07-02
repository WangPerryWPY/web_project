from http.client import HTTPResponse
from django.http import HttpResponse, HttpResponseRedirect
from .models import Question
from django.template import loader
import numpy as np
from .load_model import predict

def index(request):
  latest_question_list = Question.objects.order_by('-pub_date')
  template = loader.get_template('app/index.html')
  context = {
      'latest_question_list': latest_question_list,
  }
  return HttpResponse(template.render(context, request))

def submit(request):
  age = int(request.POST['age'])
  gender = int(request.POST['gender'])
  bmi = float(request.POST['bmi'])
  duration_of_disease = float(request.POST['duration_of_disease'])
  mc = float(request.POST['mc'])
  acth = float(request.POST['acth'])
  ufc = float(request.POST['24h_ufc'])
  k = float(request.POST['k'])
  hddst = float(request.POST['hddst'])
  lddst = float(request.POST['lddst'])
  mri = float(request.POST['mri'])
  x = np.array([age, gender, bmi, duration_of_disease, mc, acth, ufc, k, hddst, lddst, mri]).reshape(1, 11)
  pred_class, prob = predict(x)
  print(pred_class)
  print(prob)
  return HttpResponse("pred_class: " + str(pred_class) + "\nprob: " + str(prob))
