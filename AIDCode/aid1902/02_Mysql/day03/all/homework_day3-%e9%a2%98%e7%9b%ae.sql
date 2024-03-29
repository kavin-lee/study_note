-- homework_day3.sql
-- MySQL第三天作业

1. 修改acct表结构
  1）在acct_no上添加主键约束
  2）在acct_name, acct_type字段上添加非空约束
  3）在status字段添加默认约束，默认值为1

2. 创建客户信息表customer，包含字段有：
   cust_no		客户编号，字符串类型，最长32位，主键
   tel_no     	电话号码，字符串类型，最长32位，非空
   cust_name	客户姓名，字符串类型，最长64位，非空
   address		送货地址，字符串类型，最长128位，非空

3. 为customer添加数据，
   每个acct表中的cust_no都添加一笔

4. 查询所有贷款账户所属客户名称、电话号码
  （利用子查询实现）

5. 编写一个查询语句，查询结果包含如下字段：
  （联合查询来实现）
账号 户名 账户类型 余额 客户编号 客户电话 地址 

