def get_dimensions(lst):
    if not isinstance(lst, list):
        # 만약 리스트가 아니라면, 스칼라이므로 0을 반환
        return 0
    else:
        # 리스트의 첫 번째 요소의 차원을 가져옴
        sub_dimension = get_dimensions(lst[0])
        # 리스트의 깊이에 1을 더해서 반환
        return sub_dimension + 1

# 테스트 리스트
lst = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

# 리스트의 차원 확인
dimensions = get_dimensions(lst)
print("리스트의 차원:", dimensions)
