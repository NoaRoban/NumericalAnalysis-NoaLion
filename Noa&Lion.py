# Assignment 2
# Noa Ben-Gigi  ID:318355633
# Lion Dahan    ID:318873338


def copy_matrix(mat):
    """
    :param mat: matrix
    :return: deep copy of matrix
    """
    copy_mat = [[0] * len(mat[0]) for _ in range(len(mat))]  # creating new zero matrix
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            copy_mat[i][j] = mat[i][j]
    return copy_mat


def determinant_calculation(mat):
    """return the value of the determinant
    :param mat: matrix in size n*n
    :return: value
    """
    if len(mat) == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    determinant_val = 0
    size_of_new_mat = len(mat) - 1
    for i in range(len(mat)):
        new_mat = [[] * size_of_new_mat for _ in range(size_of_new_mat)]
        for k in range(1, len(mat)):
            for j in range(len(mat)):
                if j != i:
                    new_mat[k - 1].append(mat[k][j])  # add the right element
        if i % 2 == 0:
            # here we know that the i is even
            determinant_val += mat[0][i] * determinant_calculation(new_mat)
        else:
            # i is odd
            determinant_val -= mat[0][i] * determinant_calculation(new_mat)
    return determinant_val


def multiply_matrices(mat1, mat2):
    """
    :param mat1: the first matrix
    :param mat2: the second matrix
    :return: multiply between matrices
    """
    if len(mat1[0]) != len(mat2):
        return None
    result_mat = [[0] * len(mat2[0]) for _ in range(len(mat1))]
    for i in range(len(mat1)):  # rows
        for j in range(len(mat2[0])):  # cols
            for k in range(len(mat2)):
                result_mat[i][j] += (mat1[i][k] * mat2[k][j])
    return result_mat


def get_unit_matrix(size):
    """
    :param size: integer that described the matrix's size- the matrix is n*n size
    :return: the unit matrix in the right size
    """
    unit_mat = [[0] * size for _ in range(size)]
    for i in range(size):
        unit_mat[i][i] = 1
    return unit_mat


def compare_two_matrices(mat1, mat2):
    """
    :param mat1: matrix1
    :param mat2: matrix2
    :return: boolean value - true is mat1==mat2. else , false
    """
    if mat1 and mat2 is not None:
        for i in range(len(mat1)):
            for j in range(len(mat2)):
                if mat1[i][j] != mat2[i][j]:
                    return False
        return True
    else:
        return False


def find_elementary_matrix(mat):
    """this func find the elementary matrix in any level in order to find the reverse matrix
    :param mat: matrix
    :return: elementary matrix
    """
    elementary_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        for j in range(i):
            if mat[i][j] != 0:
                elementary_mat[i][j] = - mat[i][j] / mat[j][j]
                return elementary_mat


def elementary_matrix_U(mat):
    """
    this func find the elementary matrix in any level to find the reverse matrix
    :param mat:
    :return:
    """
    elementary_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            if mat[i][j] != 0:
                elementary_mat[i][j] = - mat[i][j] / mat[j][j]
                return elementary_mat


def unit_diagonal(mat):
    unit_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        if mat[i][i] != 1:
            unit_mat[i][i] = 1 / mat[i][i]
            return unit_mat
    return unit_mat


def build_matrix(size):
    """this func building the matrix
    :param size: the matrix is sized size*size
    :return:the new matrix
    """
    mat = [[0] * size for _ in range(size)]
    print("Please enter the matrix's numbers: ")
    for i in range(size):
        for j in range(size):
            print('entry in row: ', i + 1, ' column: ', j + 1)
            mat[i][j] = int(input())
    return mat


def build_vector_b(size):
    b = [[0] * size for _ in range(1)]
    print("Please enter the result of the experiment:(b vector)")
    for i in range(size):
        b[0][i] = int(input())
    return b


def print_matrix(A):
    """this func print that matrix
    :param A: matrix
    :return: no return value
    """
    print('\n'.join(['\t'.join(['{:4}'.format(item) for item in row])
                     for row in A]))


def add_between_matrices(mat1, mat2):
    """
    :param mat1: matrix1
    :param mat2: matrix2
    :return: mat1 + mat2
    """
    if len(mat1) != len(mat1):
        return None
    result_mat = [[0] * len(mat1) for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2)):
            result_mat[i][j] = mat1[i][j] + mat2[i][j]
    return result_mat


def get_upper_matrix(mat):
    """this func return U matrix
    :param mat: matrix
    :return: U_mat
    """
    temp = find_elementary_matrix(mat)
    while temp is not None:
        mat = multiply_matrices(temp, mat)
        temp = find_elementary_matrix(mat)
    return mat


def get_lower_matrix(mat):
    """this func return L matrix
    :param mat: matrix
    :return: L_mat
    """
    temp = find_elementary_matrix(mat)
    lower_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        for j in range(i):
            if temp is not None:
                if temp[i][j] != 0:
                    lower_mat[i][j] = -temp[i][j]
                mat = multiply_matrices(temp, mat)
                temp = find_elementary_matrix(mat)
    return lower_mat


def reverse_matrix(mat):  # without Gauss Elimination
    unit_mat = get_unit_matrix(len(mat))  # build unit matrix
    u_mat = find_elementary_matrix(mat)  # the first elementary matrix
    el_mat = copy_matrix(unit_mat)  # deep copy
    while u_mat is not None:
        mat = multiply_matrices(u_mat, mat)
        el_mat = multiply_matrices(u_mat, el_mat)
        u_mat = find_elementary_matrix(mat)
    l_mat = elementary_matrix_U(mat)
    while l_mat is not None:
        el_mat = multiply_matrices(l_mat, el_mat)
        mat = multiply_matrices(l_mat, mat)
        l_mat = elementary_matrix_U(mat)
    for i in range(len(mat)):
        if mat[i][i] != 1:
            diagonal_mat = unit_diagonal(mat)
            mat = multiply_matrices(diagonal_mat, mat)
            el_mat = multiply_matrices(diagonal_mat, el_mat)
    return el_mat


def inverse_mat(mat):
    """ return the inverse mat with Gauss Elimination
    :param mat:matrix in size n*n
    :return: inverse matrix
    """
    unit_mat = get_unit_matrix(len(mat))  # build unit matrix
    all_elementary_mat = unit_mat  # deep copy
    for i in range(len(mat)):
        u_mat = replace_line_in_matrix(mat, i)  # pivoting
        mat = multiply_matrices(u_mat, mat)
        all_elementary_mat = multiply_matrices(u_mat, all_elementary_mat)
        for j in range(i, len(mat)):
            if u_mat is not None or compare_two_matrices(u_mat, unit_mat):
                mat = multiply_matrices(u_mat, mat)
                all_elementary_mat = multiply_matrices(u_mat, all_elementary_mat)
            u_mat = find_elementary_matrix(mat)
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            l_mat = elementary_matrix_U(mat)
            if l_mat is not None:
                mat = multiply_matrices(l_mat, mat)
                all_elementary_mat = multiply_matrices(l_mat, all_elementary_mat)
    for i in range(len(mat)):
        if mat[i][i] != 1:
            diagonal_mat = unit_diagonal(mat)
            mat = multiply_matrices(diagonal_mat, mat)
            all_elementary_mat = multiply_matrices(diagonal_mat, all_elementary_mat)
    return all_elementary_mat


def replace_line_in_matrix(mat, i):
    """ if the pivot is zero than we replace lines
    :param mat: matrix
    :param i: index
    :return: the updated mat
    """
    unit_mat = get_unit_matrix(len(mat))
    max_value_index = 0
    check = False
    for j in range(i + 1, len(mat)):
        if abs(mat[j][i]) > abs(mat[i][i]):
            max_value_index = j
            check = True
    if check:
        temp = unit_mat[i]
        unit_mat[i] = unit_mat[max_value_index]
        unit_mat[max_value_index] = temp
    return unit_mat


def main():
    A = [[1, 2, 1],
         [2, 6, 1],
         [1, 1, 4]]
    b = [[5.5], [3.5], [1.5]]

    determinant_val = determinant_calculation(A)
    print("The value of  the determinant is: ", determinant_val)
    if determinant_val != 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("The value is not zero- we solve (A^-1)*b=x: ")
        print("The inverse matrix")
        print_matrix(inverse_mat(A))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("b vector: ")
        print_matrix(b)
        print("The result:(x vector):")
        print_matrix(multiply_matrices(reverse_matrix(A), b))
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("The value is zero")
        U = get_upper_matrix(A)
        print("U: ")
        print_matrix(U)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("L:")
        L = get_lower_matrix(A)
        print_matrix(L)
        print("A = LU") # just for checking
        print_matrix(multiply_matrices(L, U))


if __name__ == "__main__":
    main()
