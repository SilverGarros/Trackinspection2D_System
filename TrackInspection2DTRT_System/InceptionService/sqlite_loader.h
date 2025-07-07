#pragma once
#include <Windows.h>
#include <string>
#include <functional>

#ifndef SQLITE_TRANSIENT
#define SQLITE_TRANSIENT ((void(*)(void*))-1)
#endif

// 简化的SQLite结构和类型定义
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
typedef long long int sqlite3_int64;
typedef unsigned long long int sqlite3_uint64;
typedef sqlite3_int64 sqlite_int64;
typedef sqlite3_uint64 sqlite_uint64;
typedef int (*sqlite3_callback)(void*, int, char**, char**);

// 函数指针类型定义
typedef int (*sqlite3_open_t)(const char*, sqlite3**);
typedef int (*sqlite3_close_t)(sqlite3*);
typedef int (*sqlite3_exec_t)(sqlite3*, const char*, sqlite3_callback, void*, char**);
typedef int (*sqlite3_prepare_v2_t)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
typedef int (*sqlite3_step_t)(sqlite3_stmt*);
typedef int (*sqlite3_finalize_t)(sqlite3_stmt*);
typedef int (*sqlite3_bind_text_t)(sqlite3_stmt*, int, const char*, int, void(*)(void*));
typedef const unsigned char* (*sqlite3_column_text_t)(sqlite3_stmt*, int);
typedef int (*sqlite3_column_int_t)(sqlite3_stmt*, int);
typedef int (*sqlite3_reset_t)(sqlite3_stmt*);
typedef int (*sqlite3_clear_bindings_t)(sqlite3_stmt*);
typedef const char* (*sqlite3_errmsg_t)(sqlite3*);
typedef void (*sqlite3_free_t)(void*);
typedef int (*sqlite3_bind_null_t)(sqlite3_stmt*, int);
typedef int (*sqlite3_bind_double_t)(sqlite3_stmt*, int, double);
typedef int (*sqlite3_bind_int_t)(sqlite3_stmt*, int, int);

// SQLite API加载器类
class SQLiteLoader {
public:
    SQLiteLoader(const std::string& dllPath = "sqlite3.dll") {
        hModule = LoadLibraryA(dllPath.c_str());
        if (hModule) {
            // 加载必要的函数
            sqlite3_open_fn = (sqlite3_open_t)GetProcAddress(hModule, "sqlite3_open");
            sqlite3_close_fn = (sqlite3_close_t)GetProcAddress(hModule, "sqlite3_close");
            sqlite3_exec_fn = (sqlite3_exec_t)GetProcAddress(hModule, "sqlite3_exec");
            sqlite3_prepare_v2_fn = (sqlite3_prepare_v2_t)GetProcAddress(hModule, "sqlite3_prepare_v2");
            sqlite3_step_fn = (sqlite3_step_t)GetProcAddress(hModule, "sqlite3_step");
            sqlite3_finalize_fn = (sqlite3_finalize_t)GetProcAddress(hModule, "sqlite3_finalize");
            sqlite3_bind_text_fn = (sqlite3_bind_text_t)GetProcAddress(hModule, "sqlite3_bind_text");

            sqlite3_bind_int_fn = (sqlite3_bind_int_t)GetProcAddress(hModule, "sqlite3_bind_int");
            sqlite3_column_text_fn = (sqlite3_column_text_t)GetProcAddress(hModule, "sqlite3_column_text");
            sqlite3_column_int_fn = (sqlite3_column_int_t)GetProcAddress(hModule, "sqlite3_column_int");
            sqlite3_reset_fn = (sqlite3_reset_t)GetProcAddress(hModule, "sqlite3_reset");
            sqlite3_clear_bindings_fn = (sqlite3_clear_bindings_t)GetProcAddress(hModule, "sqlite3_clear_bindings");
            sqlite3_errmsg_fn = (sqlite3_errmsg_t)GetProcAddress(hModule, "sqlite3_errmsg");
            sqlite3_free_fn = (sqlite3_free_t)GetProcAddress(hModule, "sqlite3_free");
            sqlite3_bind_null_fn = (sqlite3_bind_null_t)GetProcAddress(hModule, "sqlite3_bind_null");
            sqlite3_bind_double_fn = (sqlite3_bind_double_t)GetProcAddress(hModule, "sqlite3_bind_double");
        }
    }

    ~SQLiteLoader() {
        if (hModule) {
            FreeLibrary(hModule);
        }
    }

    bool isLoaded() const { return hModule != NULL; }

    // 公开函数接口
    sqlite3_open_t sqlite3_open_fn = nullptr;
    sqlite3_close_t sqlite3_close_fn = nullptr;
    sqlite3_exec_t sqlite3_exec_fn = nullptr;
    sqlite3_prepare_v2_t sqlite3_prepare_v2_fn = nullptr;
    sqlite3_step_t sqlite3_step_fn = nullptr;
    sqlite3_finalize_t sqlite3_finalize_fn = nullptr;
    sqlite3_bind_text_t sqlite3_bind_text_fn = nullptr;
    sqlite3_column_text_t sqlite3_column_text_fn = nullptr;
    sqlite3_column_int_t sqlite3_column_int_fn = nullptr;
    sqlite3_reset_t sqlite3_reset_fn = nullptr;
    sqlite3_clear_bindings_t sqlite3_clear_bindings_fn = nullptr;
    sqlite3_errmsg_t sqlite3_errmsg_fn = nullptr;
    sqlite3_free_t sqlite3_free_fn = nullptr;
    sqlite3_bind_null_t sqlite3_bind_null_fn = nullptr;
    sqlite3_bind_double_t sqlite3_bind_double_fn = nullptr;
    sqlite3_bind_int_t sqlite3_bind_int_fn = nullptr;

private:
    HMODULE hModule = NULL;
};

// 全局SQLite加载器实例
inline SQLiteLoader g_sqliteLoader;