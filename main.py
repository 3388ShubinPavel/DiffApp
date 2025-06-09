import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
from typing import Union, List

# === Определение узлов синтаксического дерева ===
class Node:
    def differentiate(self, var: str) -> 'Node':
        # Абстрактный метод: дифференцирование относительно переменной var
        raise NotImplementedError

    def simplify(self) -> 'Node':
        # Упрощение выражения по умолчанию возвращает узел без изменений
        return self

    def to_infix(self) -> str:
        # Инфиксная запись (срединная нотация)
        raise NotImplementedError

    def to_prefix(self) -> str:
        # Префиксная запись (польская): оператор перед операндами
        raise NotImplementedError

    def to_postfix(self) -> str:
        # Постфиксная запись (обратная польская): оператор после операндов
        raise NotImplementedError

    def __str__(self) -> str:
        # При приведении к строке используем инфикс
        return self.to_infix()


class Variable(Node):
    def __init__(self, name: str):
        # Инициализация узла переменной с именем name
        self.name = name

    def differentiate(self, var: str) -> Node:
        # Производная x по x = 1, любой другой переменной = 0
        return Constant(1) if self.name == var else Constant(0)

    def to_infix(self) -> str:
        return self.name

    def to_prefix(self) -> str:
        return self.name

    def to_postfix(self) -> str:
        return self.name


class Constant(Node):
    def __init__(self, value: Union[int, float]):
        # Инициализация узла константы
        self.value = value

    def differentiate(self, var: str) -> Node:
        # Производная константы всегда 0
        return Constant(0)

    def to_infix(self) -> str:
        return str(self.value)

    def to_prefix(self) -> str:
        return str(self.value)

    def to_postfix(self) -> str:
        return str(self.value)


class Add(Node):
    def __init__(self, left: Node, right: Node):
        # Инициализация узла сложения: слева и справа операнды
        self.left = left
        self.right = right

    def differentiate(self, var: str) -> Node:
        # (u + v)' = u' + v'
        return Add(self.left.differentiate(var), self.right.differentiate(var))

    def simplify(self) -> Node:
        # Упрощаем потомков
        left = self.left.simplify()
        right = self.right.simplify()
        # Если оба константы — складываем значения
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value + right.value)
        # 0 + v = v, u + 0 = u
        if isinstance(left, Constant) and left.value == 0:
            return right
        if isinstance(right, Constant) and right.value == 0:
            return left
        # Иначе возвращаем новый узел
        return Add(left, right)

    def to_infix(self) -> str:
        return f"({self.left.to_infix()} + {self.right.to_infix()})"

    def to_prefix(self) -> str:
        return f"+ {self.left.to_prefix()} {self.right.to_prefix()}"

    def to_postfix(self) -> str:
        return f"{self.left.to_postfix()} {self.right.to_postfix()} +"


class Subtract(Node):
    def __init__(self, left: Node, right: Node):
        # Узел вычитания
        self.left = left
        self.right = right

    def differentiate(self, var: str) -> Node:
        # (u - v)' = u' - v'
        return Subtract(self.left.differentiate(var), self.right.differentiate(var))

    def simplify(self) -> Node:
        left = self.left.simplify()
        right = self.right.simplify()
        # Если обе константы — вычитаем их
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value - right.value)
        # u - 0 = u
        if isinstance(right, Constant) and right.value == 0:
            return left
        return Subtract(left, right)

    def to_infix(self) -> str:
        return f"({self.left.to_infix()} - {self.right.to_infix()})"

    def to_prefix(self) -> str:
        return f"- {self.left.to_prefix()} {self.right.to_prefix()}"

    def to_postfix(self) -> str:
        return f"{self.left.to_postfix()} {self.right.to_postfix()} -"


class Multiply(Node):
    def __init__(self, left: Node, right: Node):
        # Узел умножения
        self.left = left
        self.right = right

    def differentiate(self, var: str) -> Node:
        # (u*v)' = u'*v + u*v'
        return Add(
            Multiply(self.left.differentiate(var), self.right),
            Multiply(self.left, self.right.differentiate(var))
        )

    def simplify(self) -> Node:
        left = self.left.simplify()
        right = self.right.simplify()
        # Если константы, умножаем их
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value * right.value)
        # 0 * v = 0, u * 0 = 0
        if (isinstance(left, Constant) and left.value == 0) or (isinstance(right, Constant) and right.value == 0):
            return Constant(0)
        # 1 * v = v, u * 1 = u
        if isinstance(left, Constant) and left.value == 1:
            return right
        if isinstance(right, Constant) and right.value == 1:
            return left
        return Multiply(left, right)

    def to_infix(self) -> str:
        return f"({self.left.to_infix()} * {self.right.to_infix()})"

    def to_prefix(self) -> str:
        return f"* {self.left.to_prefix()} {self.right.to_prefix()}"

    def to_postfix(self) -> str:
        return f"{self.left.to_postfix()} {self.right.to_postfix()} *"


class Divide(Node):
    def __init__(self, left: Node, right: Node):
        # Узел деления
        self.left = left
        self.right = right

    def differentiate(self, var: str) -> Node:
        # (u/v)' = (u'*v - u*v') / v^2
        numerator = Subtract(
            Multiply(self.left.differentiate(var), self.right),
            Multiply(self.left, self.right.differentiate(var))
        )
        denominator = Multiply(self.right, self.right)  # v^2
        return Divide(numerator, denominator)

    def simplify(self) -> Node:
        left = self.left.simplify()
        right = self.right.simplify()
        # Если константы, делим их
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value / right.value)
        # u / 1 = u, 0 / v = 0
        if isinstance(right, Constant) and right.value == 1:
            return left
        if isinstance(left, Constant) and left.value == 0:
            return Constant(0)
        return Divide(left, right)

    def to_infix(self) -> str:
        return f"({self.left.to_infix()} / {self.right.to_infix()})"

    def to_prefix(self) -> str:
        return f"/ {self.left.to_prefix()} {self.right.to_prefix()}"

    def to_postfix(self) -> str:
        return f"{self.left.to_postfix()} {self.right.to_postfix()} /"


class Power(Node):
    def __init__(self, left: Node, right: Node):
        # Узел возведения в степень
        self.left = left
        self.right = right

    def differentiate(self, var: str) -> Node:
        # Если показатель константа: (u^n)' = n*u^(n-1)*u'
        if isinstance(self.right, Constant):
            new_exp = Constant(self.right.value - 1)
            return Multiply(
                Multiply(self.right, Power(self.left, new_exp)),
                self.left.differentiate(var)
            )
        # Общий случай для u^v: u^v*(v'ln(u)+v*u'/u)
        return Multiply(
            Power(self.left, self.right),
            Add(
                Multiply(self.right.differentiate(var), Ln(self.left)),
                Multiply(self.right, Divide(self.left.differentiate(var), self.left))
            )
        )

    def simplify(self) -> Node:
        left = self.left.simplify()
        right = self.right.simplify()
        # Если оба константы, возводим
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value ** right.value)
        # u^1 = u, u^0 = 1
        if isinstance(right, Constant) and right.value == 1:
            return left
        if isinstance(right, Constant) and right.value == 0:
            return Constant(1)
        return Power(left, right)

    def to_infix(self) -> str:
        return f"({self.left.to_infix()} ^ {self.right.to_infix()})"

    def to_prefix(self) -> str:
        return f"^ {self.left.to_prefix()} {self.right.to_prefix()}"

    def to_postfix(self) -> str:
        return f"{self.left.to_postfix()} {self.right.to_postfix()} ^"


class Ln(Node):
    def __init__(self, arg: Node):
        # Узел натурального логарифма
        self.arg = arg

    def differentiate(self, var: str) -> Node:
        # (ln(u))' = u'/u
        return Divide(self.arg.differentiate(var), self.arg)

    def simplify(self) -> Node:
        # Упрощаем аргумент
        return Ln(self.arg.simplify())

    def to_infix(self) -> str:
        return f"ln({self.arg.to_infix()})"

    def to_prefix(self) -> str:
        return f"ln {self.arg.to_prefix()}"

    def to_postfix(self) -> str:
        return f"{self.arg.to_postfix()} ln"


class Sin(Node):
    def __init__(self, arg: Node):
        # Узел синуса
        self.arg = arg

    def differentiate(self, var: str) -> Node:
        # (sin(u))' = cos(u)*u'
        return Multiply(Cos(self.arg), self.arg.differentiate(var))

    def simplify(self) -> Node:
        # Упрощаем аргумент
        return Sin(self.arg.simplify())

    def to_infix(self) -> str:
        return f"sin({self.arg.to_infix()})"

    def to_prefix(self) -> str:
        return f"sin {self.arg.to_prefix()}"

    def to_postfix(self) -> str:
        return f"{self.arg.to_postfix()} sin"


class Cos(Node):
    def __init__(self, arg: Node):
        # Узел косинуса
        self.arg = arg

    def differentiate(self, var: str) -> Node:
        # (cos(u))' = -sin(u)*u'
        return Multiply(
            Constant(-1),
            Multiply(Sin(self.arg), self.arg.differentiate(var))
        )

    def simplify(self) -> Node:
        return Cos(self.arg.simplify())

    def to_infix(self) -> str:
        return f"cos({self.arg.to_infix()})"

    def to_prefix(self) -> str:
        return f"cos {self.arg.to_prefix()}"

    def to_postfix(self) -> str:
        return f"{self.arg.to_postfix()} cos"


# === Парсер рекурсивным спуском ===
class Parser:
    TOKEN_PATTERN = re.compile(
        r"(?P<NUMBER>\d+(?:\.\d+)?)|"  # числа, включая десятичные
        r"(?P<VAR>[a-zA-Z]+)|"            # переменные и имена функций
        r"(?P<OP>[+\-*/^])|"             # операторы
        r"(?P<LPAREN>\()|"               # левая скобка
        r"(?P<RPAREN>\))"                # правая скобка
    )

    def __init__(self):
        # Подготовка состояний парсера
        self.tokens: List[str] = []  # список токенов выражения
        self.pos = 0  # текущая позиция
        self.current: Union[str, None] = None  # текущий токен

    def tokenize(self, expr: str) -> List[str]:
        # Разбиваем входную строку на отдельные токены
        return [m.group() for m in self.TOKEN_PATTERN.finditer(expr)]

    def parse(self, expr: str) -> Node:
        # Основной метод: очищаем пробелы, токенизируем и запускаем разбор выражения
        self.tokens = self.tokenize(expr.replace(' ', ''))
        self.pos = 0
        self.current = self.tokens[0] if self.tokens else None
        return self._expr()

    def _advance(self):
        # Переход к следующему токену в списке
        self.pos += 1
        self.current = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _expr(self) -> Node:
        # Обработчик операций '+' и '-' (наименьший приоритет)
        node = self._term()
        while self.current in ('+', '-'):
            op = self.current
            self._advance()
            right = self._term()
            node = Add(node, right) if op == '+' else Subtract(node, right)
        return node

    def _term(self) -> Node:
        # Обработчик '*' и '/'
        node = self._factor()
        while self.current in ('*', '/'):
            op = self.current
            self._advance()
            right = self._factor()
            # Создаем узел умножения или деления
            node = Multiply(node, right) if op == '*' else Divide(node, right)
        return node

    def _factor(self) -> Node:
        # Для унификации просто возвращаем уровень Power
        return self._power()

    def _power(self) -> Node:
        # Обработка '^' (правоассоциативно)
        node = self._atom()
        if self.current == '^':
            self._advance()
            exponent = self._power()
            node = Power(node, exponent)
        return node

    def _atom(self) -> Node:
        # Обработка базовых элементов: скобки, функции, числа, переменные
        tok = self.current
        # Случай скобок
        if tok == '(':
            self._advance()
            node = self._expr()
            if self.current != ')':
                raise ValueError("Ожидалась закрывающая скобка")
            self._advance()
            return node
        # Функции sin, cos, ln
        if tok in ('sin', 'cos', 'ln'):
            func = tok
            self._advance()
            if self.current != '(':
                raise ValueError(f"После функции {func} ожидается '('")
            self._advance()  # пропускаем '('
            arg = self._expr()  # парсим аргумент функции
            if self.current != ')':
                raise ValueError(f"Не хватает ')' для функции {func}")
            self._advance()
            # Возвращаем соответствующий узел функции
            if func == 'sin':
                return Sin(arg)
            if func == 'cos':
                return Cos(arg)
            return Ln(arg)
        # Числовой литерал
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            value = float(tok)
            self._advance()
            return Constant(value)
        # Переменная
        if re.fullmatch(r"[a-zA-Z]+", tok):
            name = tok
            self._advance()
            return Variable(name)
        # Если ничего не подошло — ошибка
        raise ValueError(f"Неожиданный токен: {tok}")


# === Графический интерфейс на Tkinter ===
class DiffApp:
    def __init__(self, root):
        # Настраиваем окно приложения
        root.title("Символьный дифференциатор")
        # Метка с инструкцией
        tk.Label(root, text="Введите функцию f(x):").pack(pady=5)
        # Поле ввода выражения
        self.entry = tk.Entry(root, width=50)
        self.entry.pack()
        # Кнопка запуска вычисления
        tk.Button(root, text="Вычислить", command=self.compute).pack(pady=5)
        # Окно вывода результатов с прокруткой
        self.output = scrolledtext.ScrolledText(root, width=70, height=12)
        self.output.pack()

    def compute(self):
        # Метод-обработчик нажатия кнопки
        expr = self.entry.get().strip()  # получаем введенное выражение и убираем пробелы
        if not expr:
            # Если строка пуста — показываем ошибку
            messagebox.showerror("Ошибка", "Введите выражение")
            return
        try:
            # Парсим выражение в дерево
            parser = Parser()
            tree = parser.parse(expr)
            # Получаем производную и упрощаем её
            derivative = tree.differentiate('x').simplify()
            # Формируем строки вывода
            lines = [
                f"Исходное выражение: {expr}",
                f"Дерево (инфикс): {tree.to_infix()}",
                f"Префикс: {tree.to_prefix()}",
                f"Постфикс: {tree.to_postfix()}",
                f"Производная: {derivative.to_infix()}"
            ]
            # Очищаем окно вывода и вставляем результат
            self.output.delete('1.0', tk.END)
            self.output.insert(tk.END, "\n".join(lines))
        except Exception as e:
            # При ошибке парсинга или вычислений — показываем сообщение
            messagebox.showerror("Ошибка", str(e))


if __name__ == '__main__':
    # Точка входа: создаём окно и запускаем приложение
    root = tk.Tk()
    DiffApp(root)
    root.mainloop()
