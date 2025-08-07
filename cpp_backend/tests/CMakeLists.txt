# 测试目录的CMakeLists.txt

# 启用测试
enable_testing()

# 查找Google Test (如果可用)
find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "Found Google Test, building tests")
    set(ENABLE_GTEST ON)
    
    # 创建测试可执行文件
    add_executable(test_modbus_server test_modbus_server.cpp)
    
    # 设置编译定义
    target_compile_definitions(test_modbus_server PRIVATE
        ENABLE_GTEST
        $<$<BOOL:${ENABLE_MODBUS}>:ENABLE_MODBUS>
    )
    
    # 链接依赖库 (使用库目标而不是可执行文件)
    target_link_libraries(test_modbus_server
        bamboo_cut_lib
        GTest::GTest
        GTest::Main
        ${OpenCV_LIBS}
    )
    
    # 添加测试
    add_test(NAME ModbusServerTest COMMAND test_modbus_server)
    
    # 设置测试输出目录 (只有在测试被创建时才设置)
    set_tests_properties(ModbusServerTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
else()
    message(WARNING "Google Test not found, building tests without framework")
    set(ENABLE_GTEST OFF)
    
    # 如果没有Google Test，创建一个简单的测试可执行文件
    add_executable(test_modbus_server test_modbus_server.cpp)
    
    # 设置编译定义
    target_compile_definitions(test_modbus_server PRIVATE
        $<$<BOOL:${ENABLE_MODBUS}>:ENABLE_MODBUS>
    )
    
    # 链接依赖库 (使用库目标而不是可执行文件)
    target_link_libraries(test_modbus_server
        bamboo_cut_lib
        ${OpenCV_LIBS}
    )
    
    message(STATUS "Created test executable without Google Test framework")
    
    # 注意：不添加测试，因为Google Test不可用
    # 不调用 set_tests_properties，因为没有测试被创建
endif() 