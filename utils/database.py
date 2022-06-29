import pymysql


class MariaDB():
    def __init__(self, args):
        self.ip = args.ip
        self.port = args.port
        self.user = args.user
        self.passwd = args.passwd
        self.db_name = args.db_name
        self.charset = args.charset
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = pymysql.Connect(
                host=str(self.ip),
                port=self.port,
                user=self.user,
                password=self.passwd,
                database=self.db_name,
                charset=self.charset
            )
            self.cursor = self.conn.cursor()
        except:
            raise Exception("MariaDB connection failed!")

    def execute(self, cmd):
        if self.cursor is not None:
            try:
                # 执行sql语句
                self.cursor.execute(cmd)
                # 执行sql语句
                self.conn.commit()
            except:
                # 发生错误时回滚
                self.conn.rollback()


    def close(self):
        if self.cursor is not None and self.conn is not None:
            self.cursor.close()
            self.conn.close()


