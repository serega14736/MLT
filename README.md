# Непараметрическая	регрессия.
## Формула	Надарая – Ватсона
### Постановка задачи
Пусть задано пространство объектов ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/1.PNG) и множество возможных ответов ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/2.PNG). Существует неизвестная зависимость ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/3.PNG) значения которой известны только на объектах обучающией выборки ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/4.PNG),требуется построить алгоритм ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/5.PNG), аппроксимирующий неизвестную зависимость ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/6.PNG).
### Формула Надарая-Ватсона
Будем рассматривать случай ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/2.PNG). Чтобы вычислить значение ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/7.PNG) для произвольного ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/8.PNG), воспользуемся
методом наименьших квадратов: ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/9.PNG).
Зададим веса ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/10.PNG) обучающих объектов так, чтобы они убывали по мере увеличения расстояния. Для этого введём невозрастающую, гладкую, ограниченную
функцию, называемую ядром: ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/11.PNG).
Параметр ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/12.PNG) называется шириной ядра или шириной окна сглаживания. Чем меньше ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/12.PNG), тем быстрее будут убывать веса ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/13.PNG) по мере удаления ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/14.PNG) от ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/15.PNG).
Приравняв нулю производную ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/16.PNG), получим формулу ядерного сглаживания Надарая–Ватсона: ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/17.PNG).
### Выбор ядра и ширины окна
Выбор ядра мало влияет на точность аппроксимации, но определяющим образом влияет на степень гладкости функции ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/18.PNG). Для ядерного сглаживания чаще всего берут гауссовское ядро 
![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/18.PNG) или квартическое ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/19.PNG).

![Пример ядер](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/1.PNG).
