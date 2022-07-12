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
  if pred_class.size != 1:
    return HttpResponse("Invalid Input!")
  if prob.size != 2:
    return HttpResponse("Invalid Input!")
  pred_value = pred_class[0]
  prob_eas = prob[0][0]
  prob_eas_rate = '%.2f%%' % (prob_eas * 100)
  prob_cd = prob[0][1]
  prob_cd_rate = '%.2f%%' % (prob_cd * 100)
  message = ""
  if pred_value == 1:
    message = "The diagnostic tendency for this patient is CD. The probability of diagnosing with EAS is " + prob_eas_rate + ". The probability of diagnosing with CD is " + prob_cd_rate + ".\n"
    message += "该患者的机器学习模型诊断倾向是CD。他/她患有EAS的可能性为 " + prob_eas_rate + "，患有CD的可能性为" + prob_cd_rate + "。\n"
  else:
    message = "The diagnostic tendency for this patient is EAS. The probability of diagnosing with EAS is " + prob_eas_rate + ". The probability of diagnosing with CD is " + prob_cd_rate + ".\n"
    message += "该患者的机器学习模型诊断倾向是EAS。他/她患有EAS的可能性为 " + prob_eas_rate + "，患有CD的可能性为" + prob_cd_rate + "。\n"

  message += "\nNote: This diagnostic tendency is for reference only, please make the final decision based on the specific clinical situation. The model developer is not responsible for the clinical diagnosis.\n";
  message += "请注意：该诊断倾向仅作参考，最终诊断须由医生依据具体临床情况决定。该诊断模型的开发者团队并不对具体临床诊断负责。\n";

  return HttpResponse(message)

