# 测试目录的CMakeLists.txt

# 启用测试
enable_testing()

# 查找Google Test (如果可用)
find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "Found Google Test, building tests")
    
    # 创建测试可执行文件
    add_executable(test_modbus_server test_modbus_server.cpp)
    
    # 链接依赖库
    target_link_libraries(test_modbus_server
        bamboo_cut_backend
        GTest::GTest
        GTest::Main
        ${OpenCV_LIBS}
    )
    
    # 添加测试
    add_test(NAME ModbusServerTest COMMAND test_modbus_server)
    
else()
    message(WARNING "Google Test not found, skipping tests")
    
    # 如果没有Google Test，创建一个简单的测试可执行文件
    add_executable(test_modbus_server test_modbus_server.cpp)
    
    # 链接依赖库
    target_link_libraries(test_modbus_server
        bamboo_cut_backend
        ${OpenCV_LIBS}
    )
    
    message(STATUS "Created test executable without Google Test framework")
endif()

# 设置测试输出目录
set_tests_properties(ModbusServerTest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
) 