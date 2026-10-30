#ifndef LOGGER_H
#define LOGGER_H

void set_log_file(const char* path);

void log_info(const char* message, ...);
void log_warning(const char* message, ...);
void log_error(const char* message, ...);
void log_event(const char* event_type, const char* event_details);
void log_message(const char* message, ...);

#endif // LOGGER_H