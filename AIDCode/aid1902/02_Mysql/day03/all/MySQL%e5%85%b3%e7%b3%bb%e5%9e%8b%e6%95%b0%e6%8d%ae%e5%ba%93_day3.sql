�γ̣�MySQL��ϵ�����ݿ�
���ȣ�day3

���������

�߼���ѯ
1. �Ӳ�ѯ
1�����壺һ����ѯ�����Ƕ����һ����ѯ
2�����磺
   -- ��ѯ����ƽ��ֵ�Ķ���
   SELECT * FROM orders WHERE amt > 
   (SELECT AVG(amt) FROM orders);

   -- ��ѯ����������D��ͷ�Ŀͻ��µĶ���
   SELECT * FROM orders WHERE cust_id IN
   (SELECT DISTINCT cust_id 
     FROM customer
    WHERE cust_name LIKE 'D%');
3��˵����
  - �����еĲ��ֳ�Ϊ�Ӳ�ѯ
  - �Ӳ�ѯ���Է���һ��ֵ��Ҳ���Զ��ֵ
    ��������ѯ��Ҫ��������
  - ��ִ���Ӳ�ѯ�����Ӳ�ѯ�Ľ������Ϊ���
    ��ѯ����������ִ������ѯ
  - �Ӳ�ѯִֻ��һ��
4��ʹ���Ӳ�ѯ�������һ������޷��������
   ���߲������ѯ�������ʹ���Ӳ�ѯ

5��ʾ������ѯ����û���¹������Ŀͻ���Ϣ
   ��ʾ�������������ҳ����пͻ����
         �ٴӿͻ�����ȥ����һ���ֿͻ�
   SELECT * FROM customer 
   WHERE cust_id NOT IN (
     SELECT DISTINCT cust_id FROM orders
   );

2. ���ϲ�ѯ
1��ʲô�����ϲ�ѯ��Ҳ�����Ӳ�ѯ����������е�
   ���ݽ������ӣ��õ�һ����ѯ�����
2��ʲô�����ʹ�����ϲ�ѯ������һ�����޷���ѯ
   ��������Ҫ������ʱ��ʹ�����ϲ�ѯ
   ǰ�᣺���ϵı�֮��һ��Ҫ���߼��ϵĹ�����
   SELECT a.order_id, a.amt,
          b.cust_name, b.tel_no
   FROM orders a, customer b
   WHERE a.cust_id = b.cust_id;

3���ѿ����������ϲ�ѯ���������ݣ�
  - ʲô�ǵѿ��������������ϵĳ˻�������һ��
    �µļ��ϡ���ʾ�����������еĿ��ܵ�������
  - �ѿ������͹�ϵ���ѿ������У�ȥ��û������
    �򲻴��ڵ���ϣ����ǹ�ϵ���淶�Ķ�ά��

4�����Ӳ�ѯ
  - ������(INNER Join)��û�й����������ݲ���ʾ    
   ʾ������ѯ������š����ͻ����ơ��ͻ��绰
   -- ��ʽһ��where������������
   SELECT a.order_id, a.amt,
          b.cust_name, b.tel_no
   FROM orders a, customer b
   WHERE a.cust_id = b.cust_id;

   -- ��ʽ��������Inner join�ؼ���
   SELECT a.order_id, a.amt,
          b.cust_name, b.tel_no
   FROM orders a INNER JOIN customer b
   ON a.cust_id = b.cust_id;

  - ������(OUTER Join)��û�й�����������ҲҪ
    ��ʾ�������
  �����ӣ������Ϊ��׼���ұ�����ݽ��й���
          �������ȫ����ʾ���ұ��е��ֶ�
		  ���û�й�����������ʾNULL
          LEFT JOIN ʵ��
   SELECT a.order_id, a.amt,
          b.cust_name, b.tel_no
   FROM orders a LEFT JOIN customer b
   ON a.cust_id = b.cust_id;

  �����ӣ����ұ�Ϊ��׼���������ݽ��й���
          �ұ�����ȫ����ʾ������е��ֶ�
		  ���û�й�����������ʾNULL
		  right JOIN ʵ��
   SELECT a.order_id, a.amt,
          b.cust_name, b.tel_no
   FROM orders a right JOIN customer b
   ON a.cust_id = b.cust_id;

������ϰ��
1. ����һ��������ϸ��(orders_detail), ����
���ֶ��У�
   �������   order_id   �ַ��� 32�ַ�
   ��Ʒ���   product_id �ַ��� 32�ַ�
   ��Ʒ���   amt        ������ 2С��

2. ����Ʒ��ϸ���в���3������(ע���߼�������)

3. ������������ϲ�ѯ����ѯ���Ϊ��
������� �µ����� �ͻ����� ��Ʒ��� ��Ʒ���

SQL��䣺
  CREATE TABLE orders_detail (
    order_id VARCHAR(32),
	product_id VARCHAR(32),
	amt DECIMAL(16,2) );
  
  INSERT INTO orders_detail VALUES
  ('201801010001', 'P0001', 80),
  ('201801010001', 'P0002', 20),
  ('201801010003', 'P0003', 200);

  SELECT a.order_id, a.order_date,
         b.cust_name, 
		 c.product_id, c.amt
  FROM orders a, customer b, orders_detail c
  WHERE a.cust_id = b.cust_id
  AND a.order_id = c.order_id;


Լ��(Constraint)
1.ʲô��Լ�������ݱ�����ѭ�Ĺ���
2.Ŀ�ģ���֤����һ���ԡ�������
  �����ݿ��������ݽ��С����족
3.����
1���ǿ�Լ��
  - ���壺Not null��Ҫ���ֶε�ֵ����Ϊ��
  - �﷨��
    �ֶ�����  ����(����)  NOT null

2��ΨһԼ��
  - ���壺unique���ֶε�ֵ�����ظ�
  - �﷨��
    �ֶ�����  ����(����) unique

3������(PRIMARY KEY, ��дPK)���ǿա�Ψһ
  - ���壺�����ڱ���Ψһ��ʶ������һ��ʵ��
          �ǿա�Ψһ
  - �﷨��
    �ֶ�����  ����(����) PRIMARY KEY
  
  ʾ����
  CREATE TABLE t1(
    stu_no VARCHAR(32) PRIMARY key, -- ����
	stu_name VARCHAR(32) NOT null,  -- �ǿ�
	id_card_no VARCHAR(32) UNIQUE   -- Ψһ
  );
  -- ��������
  INSERT INTO t1 
  VALUES('001','Jerry','513822199001011111');
  -- ����stu_nameΪ�����ݣ�����
  INSERT INTO t1 
  VALUES('002', NULL,'51382219900102222');
  -- ����id_card_no�ظ������ݣ�����
  INSERT INTO t1 
  VALUES('003','Tom','513822199001011111');
  -- ����stu_no�����ֵ������
  INSERT INTO t1(stu_no, stu_name)
  VALUES(NULL, 'Dekie');
  -- ����stu_noֵ�ظ�������
  INSERT INTO t1(stu_no, stu_name)
  VALUES('001','Dokas');

4��Ĭ��ֵ(DEFAULT constraint)
  - ���壺ָ��ĳ���ֶε�Ĭ��ֵ������²���
    һ������û�жԸ��ֶθ�ֵ��ϵͳ���Զ�
	����һ��Ĭ��ֵ
  - �﷨��
    �ֶ����� ����(����) DEFAULT ֵ

5���Զ�����(auto_increment)
  - ���壺���ֶα�����Ϊ������ʱ������ʱ
    ����Ҫ��ֵ��ϵͳ��ԭ���ֵ�Ļ�����
	�Զ���1��Ҫ��Ҫ������ֶα�����PK��
	��������uniqueԼ����
  - �﷨��
    �ֶ����� ����(����) auto_increment

  ʾ����
  CREATE TABLE t2(
    id INT PRIMARY KEY auto_increment,
    name VARCHAR(32),
	status INT DEFAULT 0
  );
  INSERT INTO t2 VALUES(NULL,'Jerry',1);
  INSERT INTO t2 VALUES(NULL,'Tom',2);
  INSERT INTO t2(id, name) 
  VALUES(NULL,'Hennry');  

6�����Լ����Foreign Key, ���FK��
  - ʲô�������һ��Լ�������������ǰ���ǣ�
    ĳ���ֶ��ڵ�ǰ���в���PK�����������
	��Ҳ��Ϊ�������������
  - ���ã���֤�����յ�ʵ��һ������
    �����յ������ԣ�
  - �ֶα��������Լ����Ӱ���У�
    ������һ��������в����ڵ�ʵ���ǣ�����
	��ɾ��������Ѿ������յ�ʵ�壬����
  - ��������﷨��
    �ֶ����� ����(����) ,
    -- �����ֶζ�����ɺ�
	CONSTRAINT ������� FOREIGN KEY(��ǰ���ֶ�)
	REFERENCES ���(�ֶ���)
  - ʾ����
    ����course(�γ���Ϣ��, ����course_id)
	����teacher(��ʦ��Ϣ������course_id��
	    ��course_id�ֶδ������)
CREATE TABLE course(
  course_id VARCHAR(32) PRIMARY key,
  name VARCHAR(32)
) DEFAULT charset=utf8;

CREATE TABLE teacher(
  id INT auto_increment PRIMARY key,
  name VARCHAR(32),
  course_id VARCHAR(32),
  CONSTRAINT fk_course FOREIGN KEY(course_id)
  REFERENCES course(course_id)
) DEFAULT charset=utf8;
          
  -- ��course���в���һ�ʿγ�����
  -- ��һ���γ�ʵ�壩
  INSERT INTO course 
  VALUES('0001','Python��̻���');
  -- ����teacher���в���һ������
  -- ����0001�γ�ʵ��. 
  INSERT INTO teacher 
  VALUES(1,'Jerry', '0001');  -- OK
  -- ɾ��0001�γ̣�����
  DELETE FROM course WHERE course_id='0001';
  -- ��teacher�����course_idΪ0003����Ϣ
  -- ��������0003�γ�ʵ�岻����
  INSERT INTO teacher 
  VALUES(2,'Jerry', '0003');  -- ����

  - ���ʹ�õ�����
    ��Ĵ洢�������ΪInnoDB
    ����ֶ���������б���������
	��ǰ��������ֶ����ͱ���һ��

������ϰ��ͨ���޸ı�ķ�ʽ���Լ��
7��ͨ���޸��ֶη�ʽ���Լ��
���ȣ��������Ա�t6
create table t6(  
	id int, name varchar(32),
	status int, course_id varchar(4),
	tel_no varchar(32)  );
ͨ���޸ı���������Լ��
alter table t6 add primary key(id);      -- �������
alter table t6 modify id int auto_increment;       -- ���������
alter table t6 modify status int default 0;           -- ���Ĭ��ֵ
alter table t6 modify tel_no varchar(32) 
                    unique;   -- ���ΨһԼ��
alter table t6 add CONSTRAINT fk_course_id    -- ������Լ��
FOREIGN KEY(course_id) 
REFERENCES course(course_id);

���ݵ��뵼��
1. ����
1��show variables LIKE 'secure_file%';
   �����/var/lib/mysql-files/
   ����ֻ�ܵ�������Ŀ¼
   ����ֻ�ܴӸ�Ŀ¼����
2���﷨
   select��ѯ���
   INTO outfile '�ļ�·��'
   fields terminated BY '�ֶηָ���'
   lines terminated BY '�зָ���';
3��ʾ��������orders�������е�����
   SELECT * FROM orders
   INTO outfile '/var/lib/mysql-files/orders.csv'
   fields terminated BY ','
   lines terminated BY '\n';
   -- �鿴�������ļ�(Linux Shell��ִ��)�� 
   sudo cat /var/lib/mysql-files/orders.csv

2. ����
1���﷨
  LOAD data infile '�����ļ�·��'
  INTO TABLE ������
  fields terminated BY ','
  lines terminated BY '\n';
2��ʾ��
-- ��ɾ��orders�������ݣ���ִ�е���
LOAD data infile '/var/lib/mysql-files/orders.csv'
INTO TABLE orders
fields terminated BY ','
lines terminated BY '\n';
-- ������ɺ󣬲�ѯ��ȷ��

��ĸ��ơ�������
1. ��ĸ���
  -- ��orders���ݡ���ṹȫ�����Ƶ�orders_new��
  CREATE TABLE orders_new
  SELECT * FROM orders;
  -- ��orders��ṹ���Ƶ�orders_new��
  CREATE TABLE orders_new
  SELECT * FROM orders WHERE 1=0; --��ѯ���Ϊ��
  
  ��ע���÷�ʽ���Ὣ�������Ը��Ƶ��±���

2. ���������
  -- ��orders��������Ϊorders_bak
  ALTER TABLE orders rename TO orders_bak;

  









