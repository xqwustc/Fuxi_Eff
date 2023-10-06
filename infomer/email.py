# smtplib 用于邮件的发信动作
import smtplib
# email 用于构建邮件内容
from email.mime.text import MIMEText
# 构建邮件头
from email.header import Header
import time
import sys
import base64


def send_email(addr_from, authorization_code, addr_to, smtp_server, head_from, head_to, head_subject, msg):
    time_for_email = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    # 发信方的信息：发信邮箱，QQ 邮箱授权码
    from_addr = addr_from
    password = authorization_code

    # 收信方邮箱
    to_addr = addr_to

    # 发信服务器
    smtp_server = smtp_server

    # 邮箱正文内容，第一个参数为内容，第二个参数为格式(plain 为纯文本)，第三个参数为编码
    msg = MIMEText('Time Now:{}\n {}'.format(time_for_email, msg), 'plain', 'utf-8')
    msg['From'] = Header('{} <{}>'.format(head_from,addr_from))
    msg['To'] = Header(head_to)
    msg['Subject'] = Header(head_subject)

    server = smtplib.SMTP_SSL(smtp_server)
    server.connect(smtp_server, 465)

    server.login(from_addr, password)

    server.sendmail(from_addr, to_addr, msg.as_string())
    # 关闭服务器
    server.quit()
    print('邮件已发送')


def common_send(py_name,message):
    send_email(
        addr_from='a739302733@163.com',  # 发送方邮箱
        authorization_code='TFFQPIFAEEGMLGVE',# 发送方邮箱授权码，不是密码！
        addr_to='739302733@qq.com',  # 接收方邮箱
        smtp_server='smtp.163.com',
        head_from='Server',
        head_to='Wang',
        head_subject='{}已经运行结束'.format(py_name),
        msg=message
    )