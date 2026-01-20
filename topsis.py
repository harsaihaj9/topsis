import pandas as pd
import numpy as np
import re
from io import BytesIO
from cgi import FieldStorage

def handler(request):
    try:
        form = FieldStorage(fp=request.body, environ=request.environ)

        file_item = form['file']
        weights = list(map(float, form.getvalue('weights').split(',')))
        impacts = form.getvalue('impacts').split(',')
        email = form.getvalue('email')

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return {"statusCode":400,"body":"Invalid Email"}

        if len(weights) != len(impacts):
            return {"statusCode":400,"body":"Weights and impacts mismatch"}

        for i in impacts:
            if i not in ['+','-']:
                return {"statusCode":400,"body":"Invalid impact value"}

        df = pd.read_csv(file_item.file)
        data = df.iloc[:,1:].values

        norm = data / np.sqrt((data**2).sum(axis=0))
        weighted = norm * weights

        ideal_best, ideal_worst = [], []
        for i in range(len(impacts)):
            if impacts[i] == '+':
                ideal_best.append(weighted[:,i].max())
                ideal_worst.append(weighted[:,i].min())
            else:
                ideal_best.append(weighted[:,i].min())
                ideal_worst.append(weighted[:,i].max())

        ideal_best = np.array(ideal_best)
        ideal_worst = np.array(ideal_worst)

        d_pos = np.sqrt(((weighted-ideal_best)**2).sum(axis=1))
        d_neg = np.sqrt(((weighted-ideal_worst)**2).sum(axis=1))

        score = d_neg/(d_pos+d_neg)
        df['Topsis Score'] = score
        df['Rank'] = df['Topsis Score'].rank(ascending=False)

        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        return {
            "statusCode":200,
            "headers":{
                "Content-Type":"text/csv",
                "Content-Disposition":"attachment; filename=result.csv"
            },
            "body": buffer.getvalue().decode()
        }

    except Exception as e:
        return {"statusCode":500,"body":str(e)}
