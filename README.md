# Непараметрическая	регрессия.
## Формула Надарая – Ватсона
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
![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/19.PNG) или квартическое ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/20.PNG).

![Пример ядер](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/1.png).

Выбор ширины окна ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/12.PNG) решающим образом влияет на качество восстановления зависимости. При слишком узком окне ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/21.PNG) функция ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/18.PNG) стремится пройти через все точки выборки, реагируя на шум и претерпевая резкие скачки. При слишком широком окне функция чрезмерно сглаживается и в пределе ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/22.PNG) вырождается в константу.
Чтобы оценить при данном ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/12.PNG) точность локальной аппроксимации в точке ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/14.PNG), саму эту точку необходимо исключить из обучающей выборки.Такой способ оценивания называется скользящим контролем с исключением объектов по одному (leave-one-out, LOO): ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/23.PNG) ,где минимизация осуществляется по ширине окна ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/12.PNG).

Для оценки критерия качества полученной зависимости используют сумму квадратов остатков: ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/24.PNG)

### Результаты работы реализованного метода
![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/2.png)
![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/3.png)
### SSE
![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/4.png)

## Локально взвешанное сглаживание (LOWESS)
### Постановка задачи
Оценка Надарайя–Ватсона крайне чувствительна к большим одиночным выбросам. Поэтому, на ум приходит самое простое решение — чем больше величина ошибки, тем в большей степени i-й прецедент является выбросом, и тем меньше должен быть его вес. Эти соображения приводят к идее домножить веса ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/13.PNG) на коэффициенты ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/25.PNG) где ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/26.PNG) — ещё одно ядро, вообще говоря, отличное от ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/27.PNG).

### Результаты работы метода Надарая-Ватсона с выбросом
![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/5.png)

![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/6.png)
### SSE
![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/7.PNG)
### Выбор ядра
Возможны различные варианты задания ядра ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/28.PNG).
Жёсткая фильтрация: строится вариационный ряд ошибок ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/29.PNG)
и отбрасывается некоторое количество t объектов с наибольшей ошибкой. Это соответствует ядру ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/30.PNG).
Мягкая фильтрация: используется квартическое ядро ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/31.PNG)
где ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/32.PNG) — медиана вариационного ряда ошибок.
Величина ошибки ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/33.PNG) вычисляется как ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/34.PNG)
### Алгоритм LOWESS
Вычислить оценки скользящего контроля на каждом объекте: ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/35.PNG), каждый раз пересчитывая коэфициенты ![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/img's/36.PNG), пока они не стабилизируются. Как правило, этот процесс сходится довольно быстро. Однако в практических реализациях имеет смысл вводить ограничение на количество итераций, как правило, это 2-4 итерации.
### Результаты работы метода Надарая-Ватсона с выбросом
![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/8.png)

![](https://raw.githubusercontent.com/IsmailovMukhammed/MLT/master/fig's/9.png)
### Заключение
Таким образом, можно сделать вывод, что локально взвешанное сглаживание показывает хорошие результаты, по сравнению с предыдущим методом, то есть заметно увеличилось качество апроксимации в случае с выбросом.
